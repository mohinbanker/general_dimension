import os
from os import environ

import dj_database_url
from boto.mturk import qualification

import otree.settings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# the environment variable OTREE_PRODUCTION controls whether Django runs in
# DEBUG mode. If OTREE_PRODUCTION==1, then DEBUG=False
os.environ["OTREE_PRODUCTION"] = "1"
if environ.get('OTREE_PRODUCTION') not in {None, '', '0'}:
    DEBUG = True
else:
    DEBUG = False

ADMIN_USERNAME = 'admin'

# for security, best to set admin password in an environment variable
# ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')
ADMIN_PASSWORD = "cfpbrules"

# don't share this with anybody.
SECRET_KEY = '6+2i60rc+__hp-ov1@%t0z^!yo#&x^!+=ta0ndmiaj8=tp49&#'

PAGE_FOOTER = ''

# To use a database other than sqlite,
# set the DATABASE_URL environment variable.
# Examples:
# postgres://USER:PASSWORD@HOST:PORT/NAME
# mysql://USER:PASSWORD@HOST:PORT/NAME

DATABASES = {
    'default': dj_database_url.config(
        default='sqlite:///' + os.path.join(BASE_DIR, 'db.sqlite3')
    )
    # 'default': {
    #     'ENGINE': 'django.db.backends.postgresql_psycopg2',
    #     'NAME': 'postgres',
    #     'USER': 'otree_user',
    #     'PASSWORD': 'Pr3te$ting',
    #     'HOST': 'localhost',
    #     'PORT': '5432'
    # }
}

# Sentry Account Information
SENTRY_DSN = 'http://aa8f3376659444438933f0c0e9ca57c2:e6657515ec46456b9acbed3fd557dee1@sentry.otree.org/58'

# AUTH_LEVEL:
# If you are launching a study and want visitors to only be able to
# play your app if you provided them with a start link, set the
# environment variable OTREE_AUTH_LEVEL to STUDY.
# If you would like to put your site online in public demo mode where
# anybody can play a demo version of your game, set OTREE_AUTH_LEVEL
# to DEMO. This will allow people to play in demo mode, but not access
# the full admin interface.

# AUTH_LEVEL = environ.get('OTREE_AUTH_LEVEL')
AUTH_LEVEL = "STUDY"

# setting for integration with AWS Mturk
AWS_ACCESS_KEY_ID = environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = environ.get('AWS_SECRET_ACCESS_KEY')


# e.g. EUR, CAD, GBP, CHF, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True


# e.g. en, de, fr, it, ja, zh-hans
# see: https://docs.djangoproject.com/en/1.9/topics/i18n/#term-language-code
LANGUAGE_CODE = 'en'

# if an app is included in SESSION_CONFIGS, you don't need to list it here
INSTALLED_APPS = []

# SENTRY_DSN = ''

DEMO_PAGE_INTRO_TEXT = """
oTree games
"""

# from here on are qualifications requirements for workers
# see description for requirements on Amazon Mechanical Turk website:
# http://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/ApiReference_QualificationRequirementDataStructureArticle.html
# and also in docs for boto:
# https://boto.readthedocs.org/en/latest/ref/mturk.html?highlight=mturk#module-boto.mturk.qualification

mturk_hit_settings = {
    'keywords': ['easy', 'bonus', 'choice', 'study'],
    'title': 'Title for your experiment',
    'description': 'Description for your experiment',
    'frame_height': 500,
    'preview_template': 'global/MTurkPreview.html',
    'minutes_allotted_per_assignment': 60,
    'expiration_hours': 7*24,  # 7 days
    # 'grant_qualification_id': 'YOUR_QUALIFICATION_ID_HERE',# to prevent retakes
    'qualification_requirements': [
        qualification.LocaleRequirement("EqualTo", "US"),
        qualification.PercentAssignmentsApprovedRequirement("GreaterThanOrEqualTo", 50),
        qualification.NumberHitsApprovedRequirement("GreaterThanOrEqualTo", 5),
        # qualification.Requirement('YOUR_QUALIFICATION_ID_HERE', 'DoesNotExist')
    ]
}

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = {
    'real_world_currency_per_point': (1.00/550),
    'participation_fee': 5.00,
    'num_bots': 12,
    'doc': "",
    'mturk_hit_settings': mturk_hit_settings,
}

ROOMS=[
    {
        'name': 'Gettysburg',
        'display_name': 'Gettysburg Econ Lab',
        'participant_label_file': 'duopoly_rep_treat/participant_labels.txt',
        'use_secure_urls': True,
    }
]

SESSION_CONFIGS = [
    # {
    #     'name': 'dimension',
    #     'display_name': "Dimension",
    #     'num_demo_participants': 12,
    #     'app_sequence': [
    #         'dimension',
    #     ],
    # },
    {
        'name': 'survey',
        'display_name': "Survey",
        'num_demo_participants': 4,
        'use_browser_bots' : False,
        'app_sequence': [
            'survey',
        ],
    },
    {
        'name': 'general_dimension',
        'display_name': "General Dimension",
        'num_demo_participants': 12,
        'use_browser_bots' : False,
        'app_sequence': [
            'general_dimension',
            'survey'
        ],
        'participation_fee': 10,
        'real_world_currency_per_point': (1.00/700),
        'date': "20170317",
        'time': "1430",
        'experimenter_present': True, # set false to show "Next" button on ALL pages.
    },
    # {
    #     'name': '...',
    #     'display_name': '...',
    #     'num_demo_participants': ...,
    #     'app_sequence': ['...'],
    # }

]

ROOT_URLCONF = 'general_dimension.urls'
POINTS_CUSTOM_NAME= ""

# anything you put after the below line will override
# oTree's default settings. Use with caution.
otree.settings.augment_settings(globals())

# ROOT_URLCONF = 'urls'
