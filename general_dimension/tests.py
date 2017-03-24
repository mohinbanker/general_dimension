from otree.api import Currency as c, currency_range, Submission
from . import views
from ._builtin import Bot
from .models import Constants
import random
import time

class PlayerBot(Bot):


    # NOTE: To turn on browser bots, go to settings.py and set field 'use_browser_bots' to true for both survey and duopoly_rep_treat
    def play_round(self):
        if (self.player.id_in_group == 1):
          print ("\n\n\n\n----------- ROUND " + str(self.subsession.round_number) + " ------------\n\n\n\n")
          print ("TREATMENT " + str(self.subsession.treatment) + "\n\n\n\n")
          print ("SUBSESSION ID: " + str(self.subsession.id) + "\n\n\n\n")
        # Going through instruction pages in beginning of the game
        if (self.subsession.show_instructions_base):
          yield Submission(views.IntroductionSplash, check_html = False) # Different syntax for splash pages (don't have 'next' buttons)
          yield(views.IntroductionPRA)
          yield(views.InstructionsBasics)
          yield(views.InstructionsBasicsQuiz, {"basics_q1" : "True"})
          yield(views.InstructionsRoles)
          yield(views.InstructionsRolesQuiz, {"roles_q1" : "True", "roles_q2" : "It depends"})

        # New treatment page only comes in between two treatments for paid rounds
        if ((self.subsession.show_instructions_base and self.subsession.dims > 1) or \
               (self.subsession.show_instructions_block and not self.subsession.show_instructions_base)):
          yield(views.InstructionsNewTreatment)

        if (self.subsession.show_instructions_roles):
          yield(views.InstructionsSeller)

        if (self.subsession.show_instructions_base or self.subsession.treatment_first_multiple):
          yield(views.InstructionsSellerQuiz, {"seller_q1" : "False"})

        if (self.subsession.show_instructions_roles):
          yield(views.InstructionsBuyer)
          
        if (self.subsession.show_instructions_base):
          yield(views.InstructionsBuyerQuiz, {"buyer_q1" : "test"})

        if (self.subsession.show_instructions_base):
          yield(views.InstructionsRoundResults)
          yield(views.InstructionsWaitGame)

        if (self.subsession.show_instructions_practice):
          yield(views.PracticeBegin)
        if (self.subsession.show_instructions_real):
          yield Submission(views.PracticeEnd, check_html = False)  


        # Set and submit ask prices if participant is a seller this round
        if (self.player.seller_bool):
          price_choice = random.randint(Constants.prodcost, Constants.maxprice)
          price_dims = [price_choice/(self.subsession.dims + 1)] * (self.subsession.dims + 1)
          price_stdev = 0
          self.player.create_ask(total=price_choice, pricedims=price_dims)

          yield(views.ChoiceSeller, {"ask_total" : price_choice, "ask_stdev" : price_stdev})
        
        # Randomly choose seller from available choices
        else:
          seller_choice = random.randint(1, self.subsession.sellers)
          yield(views.ChoiceBuyer, {"contract_seller_rolenum" : seller_choice})

        yield(views.RoundResults)