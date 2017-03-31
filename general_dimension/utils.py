import numpy, copy, random, math
from .models import Constants, Subsession, Player, Group
from statistics import pstdev
from . import models
from otree.export import inspect_field_names, get_field_names_for_csv, _get_table_fields
from otree.models.session import Session
from django.contrib.auth.decorators import login_required


def get_player_from_request(request):
    """
        :param request: must contain "subsession_id" "group_id" and "player_id_in_group"
        :return: player model object
    """

    result = request.POST.dict()
    print(str(result))
    subsession = Subsession.objects.get(id=result["subsession_id"])
    group = Group.objects.get(id=result["group_id"], subsession=subsession)
    player = Player.objects.get(id_in_group=int(result["player_id_in_group"]), group=group)

    return player


def export_marketdata():
    """
        TODO: profit_final mistake_count mistake_count_1 mistake_count_8 mistake_count_16
            role_count_seller role_count_seller_1 role_count_seller_8 role_count_seller_16
            role_count_buyer role_count_buyer_1 role_count_buyer_8 role_count_buyer_16
        TODO: add survey data
        key: app_label = subsession._meta.app_config.name
    """
    # this will be a list that contains data of all sessions
    body = []

    # Create the header list
    maxdim = max(Constants.treatmentdims)

    metadata_fns = ["treatmentorder", "date", "time"]
    session_fns = ['code', 'id', 'label', 'experimenter_name', 'time_scheduled', 'time_started', 'comment', 'is_demo']
    subsession_fns = get_field_names_for_csv(Subsession)
    group_fns = get_field_names_for_csv(Group)
    market_fns = get_market_headers(maxdim)
    participant_fns = ['participant_id_in_sess', 'participant_code']
    player_fns = get_field_names_for_csv(Player)
    pricedim_fns = ["p" + str(i) for i in range(1, maxdim + 1)]

    headers = session_fns + metadata_fns + subsession_fns + group_fns + market_fns + participant_fns + player_fns + \
              pricedim_fns

    # Create the body list of rows
    # get the complete result data from the database
    qs_results = models.Player.objects.select_related('subsession', 'subsession__session', 'group', 'participant') \
        .prefetch_related('ask_set') \
        .all()

    # get all sessions, order them by label
    sessions = sorted(set([p.subsession.session for p in qs_results]), key=lambda x: x.label)


    # loop through all sessions
    for session in sessions:
        session_list = list_from_obj(session_fns, session)
        metadata_list = list_from_obj(metadata_fns, session.config)

        # loop through all subsessions (i.e. rounds) ordered by round number
        subsessions = sorted(session.get_subsessions(), key=lambda x: x.round_number)
        for subsession in subsessions:
            subsession_list = list_from_obj(subsession_fns, subsession)


            # loop through all groups ordered by ID
            groups = sorted(subsession.get_groups(), key=lambda x: x.id_in_subsession)
            for group in groups:
                # group_fieldnames = group_fieldnames or get_field_names_for_csv(g.__class__)

                group_list = list_from_obj(group_fns, group)
                market_list = []
                for role in ("S1", "S2", "S3", "S4"):
                    seller = group.get_player_by_role(role)
                    market_list += [seller.id_in_group, seller.participant.id_in_session, seller.payoff,
                                    seller.ask_total, seller.ask_stdev, seller.numsold]

                    # add price dims and appropriate number of blank spaces
                    pricedims = seller.get_pricedims()
                    if len(pricedims)==0:
                        # only here if this row is not yet populated (checking data mid-stream)
                        market_list += [""]*maxdim
                    else:
                        for i in range(1, maxdim + 1):
                            if i <= subsession.dims:
                                market_list += [pricedims[i-1].value]
                            else:
                                market_list += [""]
                for role in ("B1", "B2"):
                    buyer = group.get_player_by_role(role)
                    market_list += [buyer.id_in_group, buyer.participant.id_in_session, buyer.payoff, buyer.bid_total,
                                    buyer.contract_seller_rolenum]

                # loop through all players ordered by ID
                players = sorted(group.get_players(), key=lambda x: x.participant.id_in_session)
                for player in players:
                    participant = player.participant
                    player_list = [participant.id_in_session, participant.code] + list_from_obj(player_fns, player)


                    pricedim_list = get_pd_list(player, subsession.dims, maxdim)

                    body.append(session_list + metadata_list + subsession_list + group_list + market_list +
                                player_list + pricedim_list)

    return headers, body





def get_stdev(ask_total, numdims):
    """
        Using the first experiment, estimated the stdev/max(stdev). This was necessary because in the first experiment
            players were constrained in each field to enter at most 800/numdims.  Here, we use that model, to estimate
            expected stdev utilization as a function of ask_total, and then multiple it by max(stdev) in this new
            environment.
        :param ask_total:
        :param numdims:
        :return: because these models are fit on ask_total>100, it may return a negative value for total_price<100.
            Therefore, we return the max(stdev, 1)
    """
    if ask_total == 0:
        # stupid special cases
        return (0, 0, 0)

    ask_avg = 1.* ask_total/numdims
    ask_log = math.log(ask_total)

    # if numdims == 8:
    #     stdev_util = math.exp( -0.3411935 + 0.1052575*ask_log ) - 1
    # elif numdims == 16:
    #     stdev_util = math.exp( -0.0270693 + .0509434*ask_log ) - 1
    # else:
    #     raise ValueError('{} dimensions not supported'.format(numdims))

    # Fit a linear regression so there is a common stdev_util function for all price dimensions
    stdev_util = -0.2287798 + 0.00003303819*ask_total + 0.1734626/ask_total + 0.09059366*math.log(ask_total)

    stdev_max = math.sqrt((math.pow(ask_total - ask_avg, 2) + math.pow(0 - ask_avg, 2)*(numdims - 1))/numdims)
    stdev = stdev_max*stdev_util

    return (stdev_util, stdev_max, max(stdev, 1))


def get_autopricedims(ask_total, numdims):
    """
    :param ask_total: the total price set by the seller
    :param numdims: the number of price dimensions in this treatment
    :return: dvalues: a numdims-sized list containing automatically generated dims that sum to ask_total
    """

    if ask_total < Constants.minprice or ask_total > Constants.maxprice:
        msg = 'ask total {} outside allowable range [{}, {}]'.format(ask_total, Constants.minprice, Constants.maxprice)
        raise ValueError(msg)

    # take mu and stddev from data
    mu = ask_total*1./numdims
    (stdev_util, stdev_max, stdev) = get_stdev(ask_total, numdims)

    rawvals = [0]*numdims
    # take numDim draws from a normal distribution
    for i in range(numdims):
        val = -1
        # truncated normal would be better, but we don't have scipy at the moment. This process should be equivalent
        #   (although less efficient).  It simply redraws until it gets a value in range
        while val < Constants.minprice or val > Constants.maxprice:
            val = int(round(numpy.random.normal(mu, stdev)))
        rawvals[i] = val

    # this rounds the numbers to nearest int
    #   and makes sure there is no negative #s or numbers greater than maxVal
    dvalues = copy.copy(rawvals)
    # print(dvalues)

    def diff():
        """ gets value to control while loops """
        return ask_total - sum(dvalues)

    # Now we need to get to our target value.
    # First we reduce or increase uniformly.  This should preserve the chosen stdev
    while abs(diff()) >= numdims:
        increment = int(numpy.sign(diff()))
        for dim in range(len(dvalues)):
            # increment while staying in bounds
            dvalues[dim] = max(0, dvalues[dim] + increment)
            dvalues[dim] = min(Constants.maxprice, dvalues[dim])
        # print(dvalues)

    # Now we get to our target by incrementing a randomly chosen subset of dims without replacement.
    #   This will distort the chosen stdev somewhat, but not a lot.
    #   diff() should be < numdims
    while not diff() == 0:
        # using a while statement because if a dim is already at a bound, then this may take more than one loop
        increment = int(numpy.sign(diff()))
        dims = random.sample(range(numdims), abs(diff()))
        for dim in dims:
            dvalues[dim] = max(0, dvalues[dim] + increment)
            dvalues[dim] = min(Constants.maxprice, dvalues[dim])
        # print(dvalues)

    return {
        'ask_total': ask_total,
        'ask_stdev': pstdev(dvalues),
        'stdev utilization': stdev_util,
        'stdev max': stdev_max,
        'stdev target': stdev,
        'pricedims': dvalues,
    }


def get_example_prices(sellers, dims):
    """
        example dims for instructions generated through same process as Seller's page.
        :param dims:
        :return:
    """
    # if dims == 8:
    #     s1_pd = [57, 65, 3, 1, 40, 40, 37, 82]  # 325
    #     s2_pd = [46, 91, 20, 64, 48, 45, 32, 29]  # 375
    # elif dims == 16:
    #     s1_pd = [26, 4, 2, 39, 55, 44, 34, 1, 9, 10, 29, 0, 26, 0, 23, 23]  # 325
    #     s2_pd = [7, 22, 0, 35, 16, 0, 31, 41, 28, 4, 2, 81, 32, 0, 17, 59]  # 375
    # elif dims == 1:
    #     s1_pd = [325]
    #     s2_pd = [375]
    # else:
    #     raise ValueError('{} dimensions not supported'.format(dims))
    prices = []
    for seller in sellers:
        prices.append(get_autopricedims(min(800, 325 + 50*sellers.index(seller)), dims)["pricedims"])
    return list(zip(range(1, dims + 1), zip(*prices)))