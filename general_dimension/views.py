from otree.api import Currency as c, currency_range
from . import models
from ._builtin import Page, WaitPage
from .models import Constants, PriceDim, Ask, Player, Subsession, Group, BaseSubsession
from django.http import JsonResponse, HttpResponse
from statistics import pstdev
from django.shortcuts import render
from . import utils, export
from otree.models.session import Session
from django.contrib.auth.decorators import login_required
import random


# SPLASH PAGE AND PRA
class IntroductionSplash(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_base

class IntroductionPRA(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_base


# INSTRUCTIONS PAGES
class InstructionsBasics(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_base

    def vars_for_template(self):
        return {
            'tokens_per_dollar': int(1./float(self.session.config["real_world_currency_per_point"])),
            'showup': self.session.config['participation_fee'],
        }

class InstructionsBasicsQuiz(Page):
    form_model = models.Player
    form_fields = ['basics_q1']

    def is_displayed(self):
        return self.subsession.show_instructions_base

    def vars_for_template(self):
        return {
            'tokens_per_dollar': int(1. / float(self.session.config["real_world_currency_per_point"])),
        }


class InstructionsRoles(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_base

class InstructionsRolesQuiz(Page):
    form_model = models.Player
    form_fields = ['roles_q1', 'roles_q2']

    def is_displayed(self):
        return self.subsession.show_instructions_base


class InstructionsNewTreatment(Page):

    def is_displayed(self):
        return (self.subsession.show_instructions_base and self.subsession.dims > 1) or \
               (self.subsession.show_instructions_block and not self.subsession.show_instructions_base)


class InstructionsSeller(Page):

    def vars_for_template(self):
        return {'buyers_per_group': self.subsession.buyers,
            'num_other_sellers': self.subsession.sellers-1,
            'production_cost' : Constants.prodcost,
            'price_dims': range(1, self.subsession.dims + 1),
            'seller_outcomes': range(2, self.subsession.buyers + 1)
                }

    def is_displayed(self):
        return self.subsession.show_instructions_roles

class InstructionsSellerQuiz(Page):
    form_model = models.Player
    form_fields = ['seller_q1']

    def is_displayed(self):
        return self.subsession.show_instructions_base or \
            (self.subsession.treatment_first_multiple and Constants.show_instructions_admin)


class InstructionsBuyer(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_roles

    def vars_for_template(self):
        return{
            "prices": utils.get_example_prices(range(self.subsession.sellers), self.subsession.dims),
            "sellers": range(1, self.subsession.sellers + 1)
        }

class InstructionsBuyerQuiz(Page):
    form_model = models.Player
    form_fields = ['buyer_q1']

    def is_displayed(self):
        return self.subsession.show_instructions_base


class InstructionsRoundResults(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_base

    def vars_for_template(self):
        player = Player(roledesc="Seller", payoff_marginal=225, ask_total=325, numsold=1, rolenum=1)

        choice = [1] + [0]*(self.subsession.sellers - 1)
        buyer_choices = [choice] #list(zip(range(1, 3), [[1,0],[0,1]]))
        for i in range(1, self.subsession.buyers):
            buyer_choices.append(random.sample(choice, len(choice)))
        buyer_choices = list(zip(range(1, self.subsession.buyers + 1), buyer_choices))

        return{
            "player": player,
            "subtotal": 225,
            "prices": utils.get_example_prices(range(self.subsession.sellers), self.subsession.dims),
            "prodcost": 100,
            "benefit": 325,
            "sellers": range(1, self.subsession.sellers + 1),
            "buyer_choices": buyer_choices,
            "totals": [min(325 + i * 50, Constants.maxprice) for i in range(self.subsession.sellers)]
        }

class InstructionsWaitGame(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_base


class PracticeBegin(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_practice

    def vars_for_template(self):
        otherrole = [role for role in ["Buyer", "Seller"] if role != self.player.roledesc][0]
        practicerounds = Constants.num_rounds_practice[self.subsession.block - 1]
        return {
            "otherrole": otherrole,
            "practicerounds": practicerounds
        }

class PracticeEnd(Page):

    def is_displayed(self):
        return self.subsession.show_instructions_real

    def vars_for_template(self):
        treatmentrounds = Constants.num_rounds_treatment[self.subsession.block - 1]
        return {
            "treatmentrounds": treatmentrounds,
        }




# SELLER PAGE

class ChoiceSeller(Page):

    form_model = models.Player
    form_fields = ['ask_total', 'ask_stdev']


    def is_displayed(self):
        return self.player.roledesc == "Seller"

    def vars_for_template(self):
        numpracticerounds = sum(Constants.num_rounds_practice[:self.subsession.block])
        numtreatrounds = sum(Constants.num_rounds_treatment[:self.subsession.block - 1])
        roundnum = self.subsession.round_number - numpracticerounds - numtreatrounds

        return{
            "price_dims": range(1, self.subsession.dims+1),
            "round": roundnum,
            "treatmentrounds": Constants.num_rounds_treatment[self.subsession.block - 1] 
        }

    def before_next_page(self):
        """
            If dims==1 then we need to make and aks object. For the multiple dims case, this is handled when the user
            presses the "distrubute" button or manually edits one of the dim fields.  The dim fields do not exist
            when dims==1.
        """
        if self.subsession.practiceround:
            pass
        else: 
            if self.subsession.dims == 1:
                player = self.player
                player.create_ask(player.ask_total, pricedims=[player.ask_total], auto=None, manual=None, stdev=0)




# BUYER PAGE

class ChoiceBuyer(Page):

    form_model = models.Player
    form_fields = ['contract_seller_rolenum']

    def is_displayed(self):
        return self.player.roledesc == "Buyer"

    def vars_for_template(self):
        numpracticerounds = sum(Constants.num_rounds_practice[:self.subsession.block])
        numtreatrounds = sum(Constants.num_rounds_treatment[:self.subsession.block - 1])
        roundnum = self.subsession.round_number - numpracticerounds - numtreatrounds

        # Create a list of lists where each individual list contains price dimension i for every seller           
        price_dims = []
        if self.subsession.practiceround:
            price_dims = self.participant.vars["practice_asks" + str(self.subsession.round_number)]
        else:
            for i in range(self.subsession.sellers):
                role = "S" + str(i + 1)
                price_dims.append([pd.value for pd in self.group.get_player_by_role(role).get_ask().pricedim_set.all()])

        # List where element i gives a tuple of i AND price dimension i for seller 1 to n, in order
        prices = list(zip(range(1, self.subsession.dims + 1), zip(*price_dims)))
        self.participant.vars["practice_asks" + str(self.subsession.round_number)] = price_dims if self.subsession.practiceround else []

        return {
            "prices": prices,
            "round": roundnum,
            "sellers": range(1, self.subsession.sellers + 1),
            "treatmentrounds": Constants.num_rounds_treatment[self.subsession.block - 1] 
        }

    def before_next_page(self):
        """ Create bid object.  Set buyer price attributes """
        if self.subsession.practiceround:
            self.participant.vars["practice_bids" + str(self.subsession.round_number)] = [[0] * self.subsession.sellers for i in range(self.subsession.buyers)]
            self.participant.vars["practice_bids" + str(self.subsession.round_number)][0][self.player.contract_seller_rolenum - 1] = 1
            for sellers in self.participant.vars["practice_bids" + str(self.subsession.round_number)]:
                if sum(sellers) == 0:
                    sellers[random.randint(1, self.subsession.sellers) - 1] = 1
            buyer_choices = self.participant.vars["practice_bids" + str(self.subsession.round_number)]
            items_sold = list(map(sum, zip(*buyer_choices)))[0]
            price = sum(self.participant.vars["practice_asks" + str(self.subsession.round_number)][self.player.contract_seller_rolenum - 1])
            self.player.payoff_marginal = Constants.consbenefit - price
            self.player.bid_total = price
        else:
            seller = self.group.get_player_by_role("S" + str(self.player.contract_seller_rolenum))
            ask = seller.get_ask()
            pricedims = [pd.value for pd in ask.pricedim_set.all()]

            bid = self.player.create_bid(ask.total, pricedims)

            self.group.create_contract(bid=bid, ask=ask)
            self.player.set_buyer_data()





# WAIT PAGES

class WaitStartInstructions(WaitPage):
    # This makes sures everyone has cleared the results page before the next round begins
    template_name = 'global/WaitCustom.html'

    wait_for_all_groups = True
    title_text = "Waiting for Other Participants"
    body_text = "Please wait for other participants."


class WaitStartMatch(WaitPage):
    # This makes sures everyone has cleared the instructions pages before the next round begins
    template_name = 'global/WaitCustom.html'

    wait_for_all_groups = True
    title_text = "Waiting for Other Participants"
    body_text = ""


class WaitSellersForSellers(WaitPage):
    template_name = 'global/WaitCustom.html'

    wait_for_all_groups = True
    title_text = "Waiting for Sellers"
    body_text = "You are a buyer. Please wait for the sellers to set their prices."

    def is_displayed(self):
        return self.player.roledesc == "Buyer"


class WaitBuyersForSellers(WaitPage):
    template_name = 'global/WaitCustom.html'

    wait_for_all_groups = True
    title_text = "Waiting for Sellers"
    body_text = "Please wait for the other sellers to set their prices."

    def is_displayed(self):
        return self.player.roledesc == "Seller"

    def after_all_players_arrive(self):
        # When here, sellers have all entered their prices
        # self.group.set_marketvars_seller()
        pass

class WaitGame(WaitPage):
    template_name = 'general_dimension/WaitGame.html'

    wait_for_all_groups = True

    form_model = models.Player
    form_fields = ['gamewait_numcorrect']

    def after_all_players_arrive(self):
        pass


class WaitRoundResults(WaitPage):
    # This is just a trigger to calculate the payoffs and market variables before the results page
    template_name = 'global/WaitCustom.html'

    def after_all_players_arrive(self):
        # When here, buyers and sellers have made their choices
        if self.subsession.practiceround:
            pass
        else:
            self.group.set_marketvars()


# RESULTS

class RoundResults(Page):
    def vars_for_template(self):

        price_dims = []
        totals = []
        buyer_choices = []

        if self.subsession.practiceround:
            if self.player.roledesc == "Seller":
                asks = self.participant.vars["practice_asks" + str(self.subsession.round_number)]
                if self.subsession.dims > 1:
                    asks = [[pd.value for pd in self.player.get_pricedims()]] + asks
                else:
                    asks = [[self.player.ask_total]] + asks
                if len(asks) == self.subsession.sellers:
                    self.participant.vars["practice_asks" + str(self.subsession.round_number)] = asks
                bids = self.participant.vars["practice_bids" + str(self.subsession.round_number)]
                self.player.numsold = list(map(sum, zip(*bids)))[0]
                self.player.payoff_marginal = self.player.numsold * (sum(asks[0]) - Constants.prodcost)
            price_dims = self.participant.vars["practice_asks" + str(self.subsession.round_number)]
            totals = list(map(sum, price_dims))
        else:
            # Create a list of lists where each individual list is price dimension i for all sellers
            for i in range(self.subsession.sellers):
                role = "S" + str(i + 1)
                price_dims.append([pd.value for pd in self.group.get_player_by_role(role).get_ask().pricedim_set.all()])
                totals.append(self.group.get_player_by_role(role).ask_total)

        prices = list(zip(range(1, self.subsession.dims + 1), zip(*price_dims)))

        # Matrix with rows representing buyers and columns representing sellers
        # Value of 1 at (i, j) means buyer i bought a good from seller j
        # Otherwise values are 0. 
        if self.subsession.practiceround:
            buyer_choices = self.participant.vars["practice_bids" + str(self.subsession.round_number)]
        else:
            for i in range(self.subsession.buyers):
                role = "B" + str(i + 1)
                seller_rolenum = self.group.get_player_by_role(role).contract_seller_rolenum
                buyer_choices.append([1 if i == seller_rolenum else 0 for i in range(1, self.subsession.sellers + 1)])

        buyer_choices = list(zip(range(1, self.subsession.buyers + 1), buyer_choices))


        return {
            "prices": prices,
            "subtotal": self.player.ask_total - Constants.prodcost if self.player.ask_total != None else 0,
            "prodcost": Constants.prodcost * self.player.numsold,
            "benefit": self.player.ask_total * self.player.numsold if self.player.ask_total != None else 0,
            "sellers": range(1, self.subsession.sellers + 1),
            "buyers": range(1, self.subsession.buyers + 1),
            "buyer_choices": buyer_choices,
            "totals": totals
        }


# AJAX VIEWS
# Seller Asks
def AutoPricedims(request):

    pricejson = utils.get_autopricedims(
        ask_total=int(round(float(request.POST["ask_total"]))), numdims=int(round(float(request.POST["numdims"]))))

    if not request.POST["example"] == "True":
        # If this is being called from the instructions screen, we skip adding a row
        player = utils.get_player_from_request(request)

        ask = player.create_ask(total=pricejson["ask_total"], auto=True, manual=False, stdev=pricejson["ask_stdev"],
                            pricedims=pricejson["pricedims"])

    return JsonResponse(pricejson)

def ManualPricedims(request):

    result = request.POST.dict()

    pricedims_raw = result["pricedims"].split(",")
    pricedims = [None if pd=="" else int(round(float(pd))) for pd in pricedims_raw]
    total = sum([0 if pd=="" else int(round(float(pd))) for pd in pricedims_raw])

    if not request.POST["example"] == "True":
        # If this is being called from the instructions screen, we skip adding a row
        player = utils.get_player_from_request(request)

        ask = player.create_ask(total=total, auto=False, manual=True, pricedims=pricedims)
        ask.stdev = pstdev([int(pd.value) for pd in ask.pricedim_set.all() if not pd.value==None ])
        ask.save()

        return JsonResponse({"pricedims": pricedims, "ask_total": ask.total, "ask_stdev": ask.stdev})
    else:
        # If here, this is an example request from the instructions screen
        return JsonResponse({"pricedims": pricedims, "ask_total": total, "ask_stdev": 0})



# Wait Page Game
def GameWaitIterCorrect(request):

    player = utils.get_player_from_request(request)
    player.gamewait_numcorrect += 1
    player.save()

    return JsonResponse({"gamewait_numcorrect": player.gamewait_numcorrect})



# Data Views
@login_required
def ViewData(request):
    return render(request, 'general_dimension/adminData.html', {"session_code": Session.objects.last().code})

@login_required
def AskDataView(request):
    (headers, body) = export.export_asks()

    context = {"title": "Seller Ask Data", "headers": headers, "body": body}
    return render(request, 'general_dimension/AdminDataView.html', context)

@login_required
def AskDataDownload(request):

    headers, body = export.export_asks()
    return export.export_csv("AskData", headers, body)

@login_required
def ContractDataView(request):
    (headers, body) = export.export_contracts()

    context = {"title": "Contracts Data", "headers": headers, "body": body}
    return render(request, 'general_dimension/AdminDataView.html', context)

@login_required
def ContractDataDownload(request):
    headers, body = export.export_contracts()
    return export.export_csv("ContractData", headers, body)

@login_required
def MarketDataView(request):
    headers, body = export.export_marketdata()
    context = {"title": "Market Data", "headers": headers, "body": body}

    return render(request, 'general_dimension/AdminDataView.html', context)

@login_required
def MarketDataDownload(request):

    headers, body = export.export_marketdata()
    return export.export_csv("MarketData", headers, body)

@login_required
def SurveyDataView(request):
    headers, body = export.export_surveydata()
    context = {"title": "Survey Data", "headers": headers, "body": body}

    return render(request, 'general_dimension/AdminDataView.html', context)

@login_required
def SurveyDataDownload(request):

    headers, body = export.export_surveydata()
    return export.export_csv("SurveyData", headers, body)

@login_required
def CombinedDataView(request):
    headers, body = export.export_combineddata()
    context = {"title": "Combined Data", "headers": headers, "body": body}

    return render(request, 'general_dimension/AdminDataView.html', context)

@login_required
def CombinedDataDownload(request):

    headers, body = export.export_combineddata()
    return export.export_csv("CombinedData", headers, body)

def CodebookDownload(request, app_label):

    headers, body = export.export_docs(app_label)
    return export.export_csv("Codebook", headers, body)


page_sequence = [
    WaitStartInstructions,
    IntroductionSplash,
    IntroductionPRA,
    InstructionsBasics,
    InstructionsBasicsQuiz,
    InstructionsRoles,
    InstructionsRolesQuiz,
    InstructionsNewTreatment,
    InstructionsSeller,
    InstructionsSellerQuiz,
    InstructionsBuyer,
    InstructionsBuyerQuiz,
    InstructionsRoundResults,
    InstructionsWaitGame,
    WaitStartMatch,
    PracticeBegin,
    PracticeEnd,
    ChoiceSeller,
    WaitSellersForSellers,  # for buyers while they wait for sellers # split in tow
    WaitBuyersForSellers,  # for buyers while they wait for sellers # split in tow
    ChoiceBuyer,
    WaitGame, # both buyers and sellers go here while waiting for buyers
    WaitRoundResults,
    RoundResults
]
