from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c, currency_range
)
from otree.db.models import Model, ForeignKey
from statistics import pstdev
from otree.models.session import Session
import numpy as np, math, random, copy

author = 'Dustin Beckett, Mohin Banker'

doc = """

"""


class Constants(BaseConstants):
    """
    INPUTS:
    REQUIRES: len(treatmentdims) == len(num_sellers) == len(num_buyers) == len(practicerounds) == len(num_rounds_treatment)
    REQUIRES: len(treatmentorder) == len(treatmentdims) where treatmentorder is found in settings.py
    REQUIRES: Number of participants to be divisible by num_sellers[i] + num_buyers[i] for all i in [1, len(num_sellers)]
    
    treatmentdims: Number of price dimensions for each treatment i.
    num_sellers: Number of seller roles for each treatment i
    num_buyers: Number of buyer roles for each treatment i
    practicerounds: Whether there should be practice rounds for treatment i
    num_rounds_treatment: Number of paid rounds for treatment i
    """

    #############################################################
    treatmentdims = [16]                                 
    num_sellers = [1]                                    
    num_buyers = [1]                                      
    practicerounds = [False]                         
    num_rounds_treatment = [10]                            
    #############################################################
    







    # OTHER PARAMETERS
    name_in_url = 'general_dimension'
    prodcost = 100
    consbenefit = 800
    maxprice = 800
    minprice = 0
    starting_tokens = maxprice
    show_seller_timer = True
    show_buyer_timer = True
    show_results_timer = True
    seller_timer = 60
    buyer_timer = 60
    results_timer = 30
    show_instructions_admin = False # set false to not show any instructions


    num_rounds_practice = []
    for i in range(len(practicerounds)):
        num_rounds_practice.append(2 * int(practicerounds[i]))
    players_per_group = None
    num_treatments = len(treatmentdims)
    num_rounds = sum(num_rounds_treatment) + sum(num_rounds_practice)

    # Checking requirements
    assert(len(treatmentdims) == len(num_sellers))
    assert(len(treatmentdims) == len(num_buyers))
    assert(len(treatmentdims) == len(practicerounds))
    assert(len(treatmentdims) == len(num_rounds_treatment))


class Subsession(BaseSubsession):
    practiceround = models.BooleanField(doc="True if subsession is a practice round")
    realround = models.BooleanField(doc="True if subsession is not a practice round")
    block = models.IntegerField(doc="The order in which the treatment was played in the session")
    dims = models.IntegerField(doc="The number of price dimensions in the treatment.")
    block_new = models.BooleanField(default=False, doc="True if round is the first in a new treatment block")
    treatment_first_singular = models.BooleanField(default=False, doc="True if block>1 and treatment==1")
    treatment_first_multiple = models.BooleanField(default=False, doc="True if block==2 and block1 treatment==1 ")
    sellers = models.IntegerField(doc = "The number of sellers in the treatment")
    buyers = models.IntegerField(doc = "The number of buyers in the treatment")
    treatment = models.IntegerField(doc = "The number of the treatment")

    show_instructions_base = models.BooleanField(doc="True if basic instructions are to be shown this round.")
    show_instructions_block = models.BooleanField(doc="True if new-block instructions are to be shown this round.")
    show_instructions_roles = models.BooleanField(doc="True if role-specific instructions are to be shown this round.")
    show_instructions_practice = models.BooleanField(doc="True if practice round specific instructions are to be shwn.")
    show_instructions_real = models.BooleanField(doc="True if real round specific instructions are to be shown.")

    def vars_for_admin_report(self):
        return {"session_code": self.session.code,
                }
    def before_session_starts(self):
        # new treatment rounds
        new_block_rounds = [sum(Constants.num_rounds_treatment[:i]) + sum(Constants.num_rounds_practice[:i]) + 1 for i in range(len(Constants.num_rounds_treatment) + 1)]

        # practice rounds
        practice_rounds = [new_block_rounds[i] + j for i in range(len(new_block_rounds) - 1) for j in range(Constants.num_rounds_practice[i])]

        # set treatment-level variables
        # Determine if this is the first round of a new block. This is also used to display new instructions
        if self.round_number in new_block_rounds:
            self.block_new = True
            self.block = new_block_rounds.index(self.round_number) + 1
        else:
            self.block_new = False
            # finds the block in which this round resides.
            for i in range(len(new_block_rounds)):
                if self.round_number > new_block_rounds[i] and self.round_number < new_block_rounds[i + 1]:
                    self.block = i + 1
                    break

        self.treatment = self.block
        # Is this a practice round?
        if self.round_number in practice_rounds:
            self.practiceround = True
            self.realround = False
        else:
            self.practiceround = False
            self.realround = True

        # store treatment number,  dims, and number of sellers
        self.dims = Constants.treatmentdims[self.block - 1]
        self.sellers = Constants.num_sellers[self.block - 1]
        self.buyers = Constants.num_buyers[self.block - 1]

        # Flag if this is the first round with either a multiple-dim treatment or a single-dim treatment
        #   this is used for instructions logic.
        prev_treatments = range(1, self.block)
        prev_dims = [Constants.treatmentdims[treatment - 1] for treatment in prev_treatments]
        if self.block_new and self.round_number > 1 and self.dims == 1 and min([99] + prev_dims) > 1:
            self.treatment_first_singular = True
        elif self.block_new and self.round_number > 1 and self.dims > 1 and max([0] + prev_dims) == 1:
            self.treatment_first_multiple = True

        # Instructions control variables
        #   Show_instructions are instructions shown whenever a new block happens
        #   ..._roles are role specific instructions shown a subset of the time
        #   ..._practice are practice round specifc instructions show a subset of the time
        self.show_instructions_base = True if self.round_number == 1 and Constants.show_instructions_admin else False
        self.show_instructions_block = True if self.block_new and Constants.show_instructions_admin else False
        self.show_instructions_roles = True if \
            (self.round_number == 1 or self.treatment_first_singular or self.treatment_first_multiple) and \
            Constants.show_instructions_admin else False
        self.show_instructions_practice = True if (self.practiceround and not self.round_number-1 in practice_rounds) \
            and Constants.show_instructions_admin else False
        self.show_instructions_real = True if Constants.show_instructions_admin and (self.realround and \
            self.round_number != 1 and (self.round_number - 1 in practice_rounds or self.treatment != self.in_round(self.round_number - 1).treatment)) \
            else False


        num_players = len(self.get_players())
        matrix = self.get_group_matrix()
        group_size = self.buyers + self.sellers

        if self.practiceround:
            new_matrix = np.array(matrix).reshape(num_players, 1).tolist()
            self.set_group_matrix(new_matrix)
        else:
            group_size = self.sellers + self.buyers
            new_matrix = np.array(matrix).reshape(num_players/group_size, group_size).tolist()
            self.set_group_matrix(new_matrix)
            self.group_randomly()

        for p in self.get_players():
            # set player roles
            p.set_role()
            # Players are sellers in the first practice round
            if self.block_new and self.practiceround:
                p.roledesc = "Seller"
                p.rolenum = 1
            # Players are buyers in the second practice round
            elif self.practiceround:
                p.roledesc = "Buyer"
                p.rolenum = 1

        # Generate random bids and asks for practice rounds
        # Code is executed before subsession starts so that players cannot refresh the page and get new bids/asks
        if self.practiceround:
            for player in self.get_players():
                player.participant.vars["practice_asks" + str(self.round_number)] = []
                player.participant.vars["practice_bids" + str(self.round_number)] = []
                if player.roledesc == "Seller":
                    asks = [] #self.participant.vars["practice_asks"]
                    for i in range(1, self.sellers):
                        ask_total = random.randint(Constants.prodcost, Constants.maxprice)
                        asks.append(get_autopricedims(ask_total, self.dims)["pricedims"])
                    player.participant.vars["practice_asks" + str(self.round_number)] = asks
                    player.participant.vars["practice_bids" + str(self.round_number)] = [[0] * self.sellers for i in range(self.buyers)]
                    for sellers in player.participant.vars["practice_bids" + str(self.round_number)]:
                        if sum(sellers) == 0:
                            sellers[random.randint(1, self.sellers) - 1] = 1
                else:
                    price_dims = []
                    for i in range(self.sellers):
                        ask_total = random.randint(Constants.prodcost, Constants.maxprice)
                        price_dims.append(get_autopricedims(ask_total, self.dims)["pricedims"])
                    player.participant.vars["practice_asks" + str(self.round_number)] = price_dims






class Group(BaseGroup):
    # The group class is used to store market-level data
    mkt_bid_avg = models.FloatField(doc="Average of all bids in a single group/market.")
    mkt_ask_min = models.IntegerField(doc="Minimum of all asks in a single group/market")
    mkt_ask_max = models.IntegerField(doc="Maximum of all asks in a single group/market")
    mkt_ask_spread = models.IntegerField(doc="Difference between the max and min asks in a single group/market")
    mkt_ask_stdev_min = models.FloatField(doc="Minimum of all asks standard deviations in a single group/market")

    def create_contract(self, bid, ask):
        # Contracts are only saved in paid rounds
        contract = Contract(bid=bid, ask=ask, group=self)
        contract.save()

    def set_marketvars(self):
        # This function is called after all buyers and sellers have made their choices
        # This function is only called in paid rounds
        contracts = Contract.objects.filter(group=self)
        asks = []
        stdevs = []
        for i in range(self.subsession.sellers):
            p = self.get_player_by_role("S" + str(i + 1))
            asks.append(p.ask_total)
            stdevs.append(p.ask_stdev)

        # Player data
        for contract in contracts:
            seller = contract.ask.player
            buyer = contract.bid.player

            seller.numsold += 1
            seller.payoff_marginal += seller.ask_total - Constants.prodcost 
            buyer.payoff_marginal = Constants.consbenefit - buyer.bid_total

            if self.subsession.practiceround:
                seller.payoff = 0
                buyer.payoff  = 0
            else:
                seller.payoff = seller.payoff_marginal
                buyer.payoff = buyer.payoff_marginal

        for player in self.get_players():
            if self.subsession.round_number == 1:
                # Give players their starting token allocation
                #   payoff_marginal ignores this
                player.payoff += Constants.consbenefit

            # Keep track of interim total payoff
            player.payoff_interim = player.participant.payoff
                
        # Market data
        self.mkt_ask_min = min(asks)
        self.mkt_ask_max = max(asks)
        self.mkt_ask_spread = self.mkt_ask_max - self.mkt_ask_min
        self.mkt_bid_avg = float(sum([c.bid.total for c in contracts])) / len(contracts)
        self.mkt_ask_stdev_min = min(stdevs)


class Player(BasePlayer):
    # Both
    rolenum = models.IntegerField(doc="The player's role number")
    roledesc = models.CharField(doc="The player's role description. E.g., 'Seller' or 'Buyer'")

    # Instruction Questions
    basics_q1 = models.CharField(doc = "Instructions quiz, basics")
    roles_q1 = models.CharField(doc = "Instructions quiz, roles 1")
    roles_q2 = models.CharField(doc = "Instructions quiz, roles 2")
    seller_q1 = models.CharField(doc = "Instructions quiz, seller")
    buyer_q1 = models.CharField(doc = "Instructions quiz, buyer")

    payoff_marginal  = models.CurrencyField(default=0, doc="Tracks player's earnings, ignoring endowments and ignoring practice-round status")
    payoff_interim = models.CurrencyField(default=0, doc="Player's earnings up to and including this round")
    buyer_bool = models.BooleanField(doc="True iff this player is a buyer in this round")
    seller_bool = models.BooleanField(doc="True iff this player is a seller in this round")

    # Seller
    ask_total = models.IntegerField(min=Constants.minprice, max=Constants.maxprice, doc="If a seller, ask total/sum")
    ask_stdev = models.FloatField(doc="If a seller, player's ask standard deviation")
    numsold = models.IntegerField(default=0, doc="If a seller, number of objects the player sold that round")

    # Buyer
    choice_number = [x + 1 for x in range(max(Constants.num_sellers))]
    choice_string = ["Seller " + str(i) for i in choice_number]
    bid_total = models.IntegerField(min=Constants.minprice, max=Constants.maxprice, doc="If a buyer, bid total/sum")
    contract_seller_rolenum = models.IntegerField(
        choices= list(zip(choice_number, choice_string)),
        widget=widgets.RadioSelect(),
        doc="If a buyer, the role number of the seller from whom the buyer purchased"
    )
    mistake_bool = models.IntegerField(doc="If a buyer, True if the buyer bought from the higher priced seller")
    mistake_size = models.IntegerField(default=0, doc="If a buyer, the size of the buyer's mistake")
    other_seller_ask_total = models.IntegerField(doc="If a buyer, the ask total of the seller from whom did not buy")
    other_seller_ask_stdev = models.FloatField(doc="If a buyer, the ask stdev of the seller from whom did not buy")

    # wait page game
    gamewait_numcorrect = models.IntegerField(default=0, doc="The number of words found by player in the word search")

    def create_bid(self, bid_total, pricedims):
        """ Creates a bid row associated with the buyer after the buyer makes his/her choice """

        bid = Bid(player=self, total=bid_total)
        bid.save()
        bid.set_pricedims(pricedims)

        return bid

    def create_ask(self, total, pricedims=None, auto=None, manual=None, stdev=None):
        """
            Creates an ask row associated with the seller
            :param total: integer total price
            :param pricedims: optional. list of integer pricedims
            :return: ask object
        """
        ask = Ask(player=self, total=total, auto=auto, manual=manual, stdev=stdev)
        ask.save()
        ask.set_pricedims(pricedims)

        return ask

    def get_ask(self):
        """ Get the latest ask row associated with this player """
        ask = self.ask_set.order_by("id").last()
        return ask

    def get_ask_pricedims(self):
        ask = self.get_ask()
        if ask == None:
            return []
        else:
            return ask.pricedim_set.all()

    def get_bid(self):
        """ Get the latest bid row associated with this player """
        bid = self.bid_set.last()
        return bid

    def get_bid_pricedims(self):
        bid = self.get_bid()
        if bid == None:
            return []
        else:
            return bid.pricedim_set.all()

    def get_pricedims(self):
        if self.roledesc == "Seller":
            return self.get_ask_pricedims()
        elif self.roledesc == "Buyer":
            return self.get_bid_pricedims()

    def set_buyer_data(self):
        """ This data is stored for analysis purposes. Payoffs set in group """
        rolenum_other = [rn + 1 for rn in range(self.subsession.sellers) if (rn + 1) != self.contract_seller_rolenum]
        seller = self.group.get_player_by_role("S" + str(self.contract_seller_rolenum))

        self.bid_total = seller.ask_total
        ask_diff = [self.bid_total - self.group.get_player_by_role("S" + str(rolenum)).ask_total for rolenum in rolenum_other]

        self.mistake_size = max([0] + ask_diff)
        self.mistake_bool = 0 if self.mistake_size <= 0 else 1

    def set_role(self):
        # Since we've randomized player ids in groups in the subsession class, we can assign role via id_in_group here
        if (self.id_in_group <= self.subsession.buyers):
            self.rolenum = self.id_in_group
            self.roledesc = "Buyer"
        else:
            self.rolenum = self.id_in_group - self.subsession.buyers
            self.roledesc = "Seller"

        if self.roledesc == "Seller":
            self.buyer_bool = False
            self.seller_bool = True
        else:
            self.buyer_bool = True
            self.seller_bool = False

    def role(self):
        return self.roledesc[0] + str(self.rolenum)


class Ask(Model):
    """ Stores details of a seller's ask """
    total = models.IntegerField(min=Constants.minprice, max=Constants.maxprice, doc="Total price across all dims")
    stdev = models.FloatField(min=0, doc="Standard deviation of price dimensions")
    auto = models.BooleanField(doc="True if ask was generated automatically by the 'distribute' button")
    manual = models.BooleanField(doc="True if ask was generated by seller manually adjusting a single price dim")
    player = ForeignKey(Player)

    def set_pricedims(self, pricedims):
        """ set through manual manipulation of fields """
        for i in range(self.player.subsession.dims):
            pd = self.pricedim_set.create(dimnum=i + 1, value=pricedims[i])
            pd.save()


class Bid(Model):
    """ Stores details of a buyer's bid. Not super useful at the moment given buyer's limited action space, but
        future-proofs the code somewhat. It also just gives a nice symmetry for how we deal with the two roles.
    """
    total = models.IntegerField(min=Constants.minprice, max=Constants.maxprice, doc="Total price across all dims")
    player = ForeignKey(Player)

    def set_pricedims(self, pricedims):
        """ set through manual manipulation of fields """
        for i in range(self.player.subsession.dims):
            pd = self.pricedim_set.create(dimnum = i + 1, value=pricedims[i])
            pd.save()


class Contract(Model):
    """ Relates a bid and an ask in a successful exchange """
    ask = ForeignKey(Ask, blank=True, null=True)
    bid = ForeignKey(Bid, blank=True, null=True)
    group = ForeignKey(Group)


class PriceDim(Model):   # our custom model inherits from Django's base class "Model"

    value = models.IntegerField(doc="The value of this price dim")
    dimnum = models.IntegerField(doc="The number of the dimension of this price dim")

    # in reality, there will be either, but not both, an ask or a bid associated with each pricedim
    ask = ForeignKey(Ask, blank=True, null=True)    # creates 1:m relation -> this decision was made by a certain seller
    bid = ForeignKey(Bid, blank=True, null=True)


# Helper function for get_autopricedims function
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
        return (0, 0, 0)

    ask_avg = 1.* ask_total/numdims
    ask_log = math.log(ask_total)

    # Fit a linear regression so there is a common stdev_util function for all price dimensions
    # The larger the ask totals, the larger the difference between the original stdev_util and the interpolated stdev_util 
    stdev_util = -0.2287798 + 0.00003303819*ask_total + 0.1734626/ask_total + 0.09059366*math.log(ask_total)

    stdev_max = math.sqrt((math.pow(ask_total - ask_avg, 2) + math.pow(ask_avg, 2)*(numdims - 1))/numdims)
    stdev = stdev_max*stdev_util

    return (stdev_util, stdev_max, max(stdev, 1))


# Function copied from utils.py to generate random sub-prices for practice rounds
# Error occurs when utils.py is imported into models.py
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
            val = int(round(np.random.normal(mu, stdev)))
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
        increment = int(np.sign(diff()))
        for dim in range(len(dvalues)):
            # increment while staying in bounds
            dvalues[dim] = max(0, dvalues[dim] + increment)
            dvalues[dim] = min(Constants.maxprice, dvalues[dim])

    # Now we get to our target by incrementing a randomly chosen subset of dims without replacement.
    #   This will distort the chosen stdev somewhat, but not a lot.
    #   diff() should be < numdims
    while not diff() == 0:
        # using a while statement because if a dim is already at a bound, then this may take more than one loop
        increment = int(np.sign(diff()))
        dims = random.sample(range(numdims), abs(diff()))
        for dim in dims:
            dvalues[dim] = max(0, dvalues[dim] + increment)
            dvalues[dim] = min(Constants.maxprice, dvalues[dim])

    return {
        'ask_total': ask_total,
        'ask_stdev': pstdev(dvalues),
        'stdev utilization': stdev_util,
        'stdev max': stdev_max,
        'stdev target': stdev,
        'pricedims': dvalues,
    }