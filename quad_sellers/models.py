from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c, currency_range
)
from otree.db.models import Model, ForeignKey
from statistics import pstdev
from otree.models.session import Session
import numpy as np


author = 'Dustin Beckett'

doc = """

"""


class Constants(BaseConstants):
    name_in_url = 'two_or_four_seller'
    players_per_group = None
    buyers_per_group = 2
    treatmentdims = [1, 1, 16, 16]
    num_sellers = [2, 4, 2, 4]
    practicerounds = [True, True, True, True]
    num_treatments = 4
    num_rounds_treatment = 1
    num_rounds_practice = 2
    num_rounds = num_rounds_treatment * num_treatments + num_rounds_practice * sum([ 1 if x else 0 for x in practicerounds ])
    num_players = 12
    prodcost = 100
    consbenefit = 800
    maxprice = 800
    minprice = 0
    starting_tokens = maxprice
    # For convenience of testing the experience of players
    show_instructions_admin = False # set false to not show any instructions whatsoever


class Subsession(BaseSubsession):
    practiceround = models.BooleanField(doc="True if subsession is a practice round")
    realround = models.BooleanField(doc="True if subsession is not a practice round")
    block = models.IntegerField(doc="The order in which the treatment was played in the session")
    treatment = models.IntegerField(doc="The number of the treatment. 1=1, 2=8, 3=16")
    dims = models.IntegerField(doc="The number of price dimensions in the treatment.")
    block_new = models.BooleanField(default=False, doc="True if round is the first in a new treatment block")
    treatment_first_singular = models.BooleanField(default=False, doc="True if block>1 and treatment==1")
    treatment_first_multiple = models.BooleanField(default=False, doc="True if block==2 and block1 treatment==1 ")
    sellers = models.IntegerField(doc = "The number of sellers in the treatment")

    show_instructions_base = models.BooleanField(doc="True if basic instructions are to be shown this round.")
    show_instructions_block = models.BooleanField(doc="True if new-block instructions are to be shown this round.")
    show_instructions_roles = models.BooleanField(doc="True if role-specific instructions are to be shown this round.")
    show_instructions_practice = models.BooleanField(doc="True if practice round specific instructions are to be shwn.")
    show_instructions_real = models.BooleanField(doc="True if real round specific instructions are to be shown.")

    def vars_for_admin_report(self):
        return {"session_code": self.session.code,
                }
    def before_session_starts(self):

        def numpracticerounds(block):
            """
                Helper function to determine treatment block variables.
                :param block: block number (1,2)
                :return: the number of practice rounds that occured before treat, including current block
            """
            nums = [Constants.num_rounds_practice if x else 0 for x in Constants.practicerounds[: block]]
            return sum(nums)

        # take the string inputted by the experimenter and change it to a list
        treatmentorder = [int(t) for t in self.session.config["treatmentorder"].split(",")]

        # new treatment rounds
        new_block_rounds = [1]
        for i in range(Constants.num_treatments):
            new_block_rounds.append(Constants.num_rounds_treatment * (i + 1) + numpracticerounds(i + 1) + 1)

        # practice rounds
        practice_rounds = []
        for r in new_block_rounds:
            for i in range(r, r + Constants.num_rounds_practice):
               practice_rounds.append(i)

        # set treatment-level variables
        # Determine if this is the first round of a new block. This is also used to display new instructions
        if self.round_number in new_block_rounds:
            self.block_new = True
            self.block = new_block_rounds.index(self.round_number) + 1
        else:
            self.block_new = False
            # finds the block in which this round resides. seems like there must be an easier way...
            self.block = new_block_rounds.index(
                min(new_block_rounds, key=lambda x: abs(self.round_number - x) if x <= self.round_number else 999)) + 1


        # Is this a practice round?
        if self.round_number in practice_rounds:
            self.practiceround = True
            self.realround = False
        else:
            self.practiceround = False
            self.realround = True

        # store treatment number,  dims, and number of sellers
        self.treatment = treatmentorder[self.block - 1]
        self.dims = Constants.treatmentdims[self.treatment - 1]
        self.sellers = Constants.num_sellers[self.treatment - 1]

        # Flag if this is the first round with either a multiple-dim treatment or a single-dim treatment
        #   this is used for instructions logic.
        prev_treatments = treatmentorder[: (self.block - 1)]
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
        self.show_instructions_real = True if (self.realround and self.round_number - 1 in practice_rounds) \
                                                  and Constants.show_instructions_admin else False

        # Set player level variables
        # Randomize groups each round.
        # If this is practice round and if previous round was a practice round,
        #    then play opposite role this round
        # if self.round_number in practice_rounds and self.round_number-1 in practice_rounds:
        #     self.group_like_round(self.round_number - 1)
        #     matrix = self.get_group_matrix()
        #     for group in matrix:
        #         # since roles assigned by row position, this should flip roles between buyer and seller
        #         group.reverse()
        #     self.set_group_matrix(matrix)
        # else:
        #     self.group_randomly()
        
        matrix = self.get_group_matrix()
        num_players = len(self.get_players())
        if (self.sellers == 4):
            # Convert to numpy array temporarily because it allows easier regrouping
            new_matrix = np.array(matrix).reshape(num_players/6, 6).tolist()
            self.set_group_matrix(new_matrix)
        else:
            new_matrix = np.array(matrix).reshape(num_players/4, 4).tolist()
            self.set_group_matrix(new_matrix)

        self.group_randomly()

        for p in self.get_players():
            # set player roles
            p.set_role()


class Group(BaseGroup):
    # The group class is used to store market-level data
    mkt_bid_avg = models.FloatField(doc="Average of all bids in a single group/market.")
    mkt_ask_min = models.IntegerField(doc="Minimum of all asks in a single group/market")
    mkt_ask_max = models.IntegerField(doc="Maximum of all asks in a single group/market")
    mkt_ask_spread = models.IntegerField(doc="Difference between the max and min asks in a single group/market")
    mkt_ask_stdev_min = models.FloatField(doc="Minimum of all asks standard deviations in a single group/market")

    def create_contract(self, bid, ask):
        contract = Contract(bid=bid, ask=ask, group=self)
        contract.save()

    def set_marketvars(self):
        # this gets hit after all buyers and sellers have made their choices
        # sellers = Player.objects.filter(group=self, roledesc="Seller")
        # buyers = Player.objects.filter(group=self, roledesc="Buyer")

        contracts = Contract.objects.filter(group=self)
        asks = [p.ask_total for p in [self.get_player_by_role("S1"), self.get_player_by_role("S2")]]
        stdevs = [p.ask_stdev for p in [self.get_player_by_role("S1"), self.get_player_by_role("S2")]]

        if (self.subsession.sellers == 4):
            asks += [p.ask_total for p in [self.get_player_by_role("S3"), self.get_player_by_role("S4")]]
            stdevs += [p.ask_stdev for p in [self.get_player_by_role("S3"), self.get_player_by_role("S4")]]

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
        # self.mkt_ask_min = min([c.ask.total for c in contracts])
        # self.mkt_ask_max = max([c.ask.total for c in contracts])
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
    basics_q1 = models.CharField()
    roles_q1 = models.CharField()
    roles_q2 = models.CharField()
    seller_q1 = models.CharField()
    buyer_q1 = models.CharField()

    payoff_marginal  = models.CurrencyField(default=0, doc="Tracks player's earnings, ignoring endowments and ignoring practice-round status")
    payoff_interim = models.CurrencyField(default=0, doc="Player's earnings up to and including this round")
    buyer_bool = models.BooleanField(doc="True iff this player is a buyer in this round")
    seller_bool = models.BooleanField(doc="True iff this player is a seller in this round")

    # Seller
    ask_total = models.IntegerField(min=Constants.minprice, max=Constants.maxprice, doc="If a seller, ask total/sum")
    ask_stdev = models.FloatField(doc="If a seller, player's ask standard deviation")
    numsold = models.IntegerField(default=0, doc="If a seller, number of objects the player sold that round")

    # Buyer
    bid_total = models.IntegerField(min=Constants.minprice, max=Constants.maxprice, doc="If a buyer, bid total/sum")
    contract_seller_rolenum = models.IntegerField(
        choices=[(1, "Seller 1"), (2, "Seller 2"), (3, "Seller 3"), (4, "Seller 4")],
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
        # if pricedims == None:
        #ask.generate_pricedims()
        # else:
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
        rolenum_other = [ rn for rn in [1,2] if rn != self.contract_seller_rolenum][0]
        seller = self.group.get_player_by_role("S" + str(self.contract_seller_rolenum))
        seller_other = self.group.get_player_by_role("S" + str(rolenum_other))

        self.bid_total = seller.ask_total
        self.other_seller_ask_total = seller_other.ask_total
        self.other_seller_ask_stdev = seller_other.ask_stdev
        self.mistake_size = max(0, self.bid_total - self.other_seller_ask_total)
        self.mistake_bool = 0 if self.mistake_size <= 0 else 1

        self.other_seller_ask_stddev = pstdev([ pd.value for pd in seller_other.get_ask().pricedim_set.all() ])


    def set_role(self):
        # since we've randomized player ids in groups in the subsession class, we can assign role via id_in_group here
        if self.id_in_group == 1:
            self.rolenum = 1
            self.roledesc = "Buyer"
        elif self.id_in_group == 2:
            self.rolenum = 2
            self.roledesc = "Buyer"
        elif self.id_in_group == 3:
            self.rolenum = 1
            self.roledesc = "Seller"
        elif self.id_in_group == 4:
            self.rolenum = 2
            self.roledesc = "Seller"
        elif self.id_in_group == 5:
            self.rolenum = 3
            self.roledesc = "Seller"
        else:
            self.rolenum = 4
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

    # def generate_pricedims(self):
    #     """ set through auto-generation of price dims """
    #     for i in range(self.player.subsession.dims):
    #
    #         # pd = PriceDim(ask=self, dimnum=i + 1)
    #         pd = self.pricedim_set.create(dimnum=i + 1)
    #         pd.save()

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
