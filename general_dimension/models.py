from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c, currency_range
)
from otree.db.models import Model, ForeignKey
from statistics import pstdev
from otree.models.session import Session
import numpy as np
import math


author = 'Dustin Beckett'

doc = """

"""


class Constants(BaseConstants):
    # INPUTS:
    # REQUIRES: len(treatmentdims) == len(num_sellers) == len(num_buyers) == len(practicerounds) == len(num_rounds_treatment)
    # REQUIRES: len(treatmentorder) == len(treatmentdims) where treatmentorder is found in settings.py
    # REQUIRES: Number of participants to be divisible by num_sellers[i] + num_buyers[i] for all i in [1, len(num_sellers)]
    # treatmentdims: Number of price dimensions for each treatment i
    # num_sellers: Number of seller roles for each treatment i
    # num_buyers: Number of buyer roles for each treatment i
    # practicerounds: Whether there should be practice rounds for treatment i
    # num_rounds_treatment: Number of paid rounds for  treatment i
    treatmentdims = [16, 16]
    num_sellers = [1, 4]
    num_buyers = [3, 4]
    practicerounds = [False, False]
    num_rounds_treatment = [2, 2]
    
    # Checking requirements
    assert(len(treatmentdims) == len(num_sellers))
    assert(len(treatmentdims) == len(num_buyers))
    assert(len(treatmentdims) == len(practicerounds))
    assert(len(treatmentdims) == len(num_rounds_treatment))


    # Calculates the minimum number of practice rounds for each participant to experience both roles of each treatment
    num_rounds_practice = []
    for i in range(len(num_sellers)):
        num_rounds_practice.append(math.ceil((num_sellers[i] + num_buyers[i])/(min(num_sellers[i], num_buyers[i]))) * int(practicerounds[i]))
    # num_rounds_practice = [math.ceil((num_sellers[i] + num_buyers[i])/(min(num_sellers[i], num_buyers[i]))) for i in range(len(num_sellers))]
    name_in_url = 'general_dimension'
    players_per_group = None
    num_treatments = len(treatmentdims)
    num_rounds = sum(num_rounds_treatment) + sum(num_rounds_practice)
    # num_players = 12
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
    buyers = models.IntegerField(doc = "The number of buyers in the treatment")

    show_instructions_base = models.BooleanField(doc="True if basic instructions are to be shown this round.")
    show_instructions_block = models.BooleanField(doc="True if new-block instructions are to be shown this round.")
    show_instructions_roles = models.BooleanField(doc="True if role-specific instructions are to be shown this round.")
    show_instructions_practice = models.BooleanField(doc="True if practice round specific instructions are to be shwn.")
    show_instructions_real = models.BooleanField(doc="True if real round specific instructions are to be shown.")
    player_practice = dict()


    def vars_for_admin_report(self):
        return {"session_code": self.session.code,
                }
    def before_session_starts(self):
        # take the string inputted by the experimenter and change it to a list
        treatmentorder = [int(t) for t in self.session.config["treatmentorder"].split(",")]

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
        self.buyers = Constants.num_buyers[self.treatment - 1]

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


        matrix = self.get_group_matrix()
        num_players = len(self.get_players())
        group_size = self.sellers + self.buyers
        # Convert to numpy array temporarily because it allows easier regrouping
        new_matrix = np.array(matrix).reshape(num_players/group_size, group_size).tolist()

        if self.block_new:
            for player in self.get_players():
                player.buyer_in_practice = False
                player.seller_in_practice = False
                player.role_in_practice = False
        else:
            for player in self.get_players():
                player_last_round = player.in_round(self.round_number - 1)
                player.buyer_in_practice = player_last_round.buyer_in_practice
                player.seller_in_practice = player_last_round.seller_in_practice
                player.role_in_practice = False

        for player in self.get_players():
            player.role_in_practice = False


        if (self.round_number-1) in practice_rounds and self.round_number in practice_rounds:
            new_matrix = []
            for i in range(int(num_players/group_size)):
                new_matrix.append([])

                if (len(new_matrix[i]) < self.buyers):
                    for player in self.get_players():
                        if not player.buyer_in_practice and not player.role_in_practice:
                            player.buyer_in_practice = True
                            player.role_in_practice = True
                            new_matrix[i].append(player)
                        if not len(new_matrix[i]) < self.buyers:
                            break

                # Adding buyers with players who have been both buyer and seller
                if len(new_matrix[i]) < self.buyers:
                    for player in self.get_players():
                        if (player.buyer_in_practice and player.seller_in_practice) and not player.role_in_practice:
                            player.role_in_practice = True
                            new_matrix[i].append(player)
                        if not len(new_matrix[i]) < self.buyers:
                            break

                # Adding in anybody to fill in leftover buyer spots
                if len(new_matrix[i]) < self.buyers:
                    for player in self.get_players():
                        if not player.role_in_practice:
                            player.role_in_practice = True
                            new_matrix[i].append(player)
                        if not len(new_matrix[i]) < self.buyers:
                            break

                # Adding sellers with players who haven't been sellers
                if len(new_matrix[i]) >= self.buyers and len(new_matrix[i]) < group_size:
                    for player in self.get_players():
                        if not player.seller_in_practice and not player.role_in_practice:
                            player.seller_in_practice = True
                            player.role_in_practice = True
                            new_matrix[i].append(player)
                        if not len(new_matrix[i]) < group_size:
                            break

                # Adding sellers with players who have been both buyer and seller
                if len(new_matrix[i]) >= self.buyers and len(new_matrix[i]) < group_size:
                    for player in self.get_players():
                        if (player.buyer_in_practice and player.seller_in_practice) and not player.role_in_practice:
                            player.role_in_practice = True
                            new_matrix[i].append(player)
                        if not len(new_matrix[i]) < group_size:
                            break

                # Adding in anybody to fill in leftover seller spots
                if len(new_matrix[i]) >= self.buyers and len(new_matrix[i]) < group_size:
                    for player in self.get_players():
                        if not player.role_in_practice:
                            player.role_in_practice = True
                            new_matrix[i].append(player)
                        if not len(new_matrix[i]) < group_size:
                            break

            print(new_matrix)
            self.set_group_matrix(new_matrix)

        else:
            self.set_group_matrix(new_matrix)
            self.group_randomly()

        for p in self.get_players():
            # set player roles
            p.set_role()
            if self.block_new:
                if p.roledesc == "Buyer":
                    p.buyer_in_practice = True
                else:
                    p.seller_in_practice = True


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

    buyer_in_practice = models.BooleanField()
    seller_in_practice = models.BooleanField()
    role_in_practice = models.BooleanField()

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
        rolenum_other = [ rn + 1 for rn in range(self.subsession.sellers) if rn != self.contract_seller_rolenum]
        seller = self.group.get_player_by_role("S" + str(self.contract_seller_rolenum))
        # seller_other = self.group.get_player_by_role("S" + str(rolenum_other))

        self.bid_total = seller.ask_total
        ask_diff = [self.bid_total - self.group.get_player_by_role("S" + str(rolenum)).ask_total for rolenum in rolenum_other]

        # self.other_seller_ask_total = seller_other.ask_total
        # self.other_seller_ask_stdev = seller_other.ask_stdev
        self.mistake_size = max([0] + ask_diff)
        self.mistake_bool = 0 if self.mistake_size <= 0 else 1

        # self.other_seller_ask_stddev = pstdev([ pd.value for pd in seller_other.get_ask().pricedim_set.all() ])


    def set_role(self):
        # since we've randomized player ids in groups in the subsession class, we can assign role via id_in_group here
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
