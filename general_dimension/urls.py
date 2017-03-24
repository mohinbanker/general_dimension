from django.conf.urls import url
from otree.urls import urlpatterns

from . import views

# Seller AJAX
urlpatterns.append(url(r'^duopoly/autopricedims/$', views.AutoPricedims, name="autopricedims"))
urlpatterns.append(url(r'^duopoly/manualpricedims/$', views.ManualPricedims, name="manualpricedims"))

# Wait game AJAX
urlpatterns.append(url(r'^duopoly/gamewaititercorrect/$', views.GameWaitIterCorrect, name="waitgame"))


# DATA
urlpatterns.append(url(r'^duopoly/data/$', views.ViewData, name="data"))

urlpatterns.append(url(r'^duopoly/data/ask/download$', views.AskDataDownload, name="ask_data_download"))
urlpatterns.append(url(r'^duopoly/data/ask$', views.AskDataView, name="ask_data_view"))

urlpatterns.append(url(r'^duopoly/data/contract/download$', views.ContractDataDownload, name="contract_data_download"))
urlpatterns.append(url(r'^duopoly/data/contract', views.ContractDataView, name="contract_data_view"))

urlpatterns.append(url(r'^duopoly/data/market/download$', views.MarketDataDownload, name="market_data_download"))
urlpatterns.append(url(r'^duopoly/data/market$', views.MarketDataView, name="market_data_view"))

urlpatterns.append(url(r'^duopoly/data/combined/download$', views.CombinedDataDownload, name="combined_data_download"))
urlpatterns.append(url(r'^duopoly/data/combined', views.CombinedDataView, name="combined_data_view"))

urlpatterns.append(url(r'^duopoly/data/survey/download$', views.SurveyDataDownload, name="survey_data_download"))
urlpatterns.append(url(r'^duopoly/data/survey', views.SurveyDataView, name="survey_data_view"))

# urlpatterns.append(url(r'^duopoly/data/codebook/download/(?P<code>[a-z0-9]+)/$', views.CodebookDownload, name="codebook_download"))
urlpatterns.append(url(r'^duopoly/data/codebook/download/(?P<app_label>[\w.]+)/$', views.CodebookDownload, name="codebook_download"))
