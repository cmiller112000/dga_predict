"""Generates data for train/test algorithms"""
from datetime import datetime
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


from bs4 import UnicodeDammit
import urllib.request
import urllib.error

from zipfile import ZipFile, BadZipfile

import pickle as pickle
import os
import random
import tldextract
import sys
import csv

from dga_classifier.dga_generators import banjori, corebot, cryptolocker, \
    dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda

# Location of Alexa 1M
#ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'
ALEXA_1M = 'http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip'

# Temporary file name, to avoid conflicts
tmpAlexaFileName = "/tmp/alexa-top1M-" + format(random.randrange(1,65535)) + ".csv"

# Logfile. Records the same output as the script
logFileName = "/tmp/alexa-ruleset-log-" + format(random.randrange(1,65535)) + ".log"

# Filename of the CSV file contained in the Alexa zipfile
tmpAlexaZipFileContents = 'top-1m.csv'

# Maximum number of websites to use in the Alexa Top 1M (i.e. it's no longer 1M but maxSitesNumber)
# Set to -1 for 'unlimited'
maxSitesNumber = 10000

# Our ourput file containg all the training data
DATA_FILE = 'traindata.pkl'
IN_DATA_FILE = 'Data_File1'
IN_HACKATHON_FILE = '/home/cher/participantTrainData_wfeatures.csv'

def get_alexa(num, address=ALEXA_1M, filename=tmpAlexaZipFileContents):
    """Grabs Alexa 1M"""
# Variables and constants
    sitesList = []
    try:
        tmpAlexaZipFileName, headers = urllib.request.urlretrieve(address)
    except urllib.error.URLError as e:
        print("Failed to download Alexa Top 1M")
        sys.exit('Error message: %s' % e)
    # Now unzip it
    try:
        # Extract in /tmp/
        print("Start extracting %s" % tmpAlexaZipFileName)
        tmpAlexaZipFile = ZipFile(tmpAlexaZipFileName,'r')
        tmpAlexaZipFile.extractall('/tmp/')
    except BadZipfile:
        sys.exit("The zip file %s is corrupted.",tmpAlexaZipFileName)

    try:
        # Rename the file to match the file with the random in it
        os.rename('/tmp/' + filename,tmpAlexaFileName)
        print("Alexa Top1M retrieved and stored in %s" % tmpAlexaFileName)
    except OSError as e:
        print("Failed to rename /tmp/top-1M.csv to %s." % (tmpAlexaFileName))
        sys.exit('Error message: %s' % (e))

    sitesReader = csv.reader(open(tmpAlexaFileName), delimiter=',', quotechar='"')
    for row in sitesReader:
        try:
            # Since some Alexa sites are not FQDNs, split where there's a "/" and keep ony the first part
            siteFQDN = sitesList.append(row[1].split("/",1)[0])
            # print("Line %s: %s" % (sitesReader.line_num, sitesList[len(sitesList) - 1])) # Outputs the current line
            if sitesReader.line_num == num:
                break
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (tmpAlexaFileName, sitesReader.line_num, e))

    retlist = [tldextract.extract(x).domain for x in sitesList ]
    return retlist

def gen_malicious(num_per_dga=maxSitesNumber):
    """Generates num_per_dga of each DGA"""
    domains = []
    labels = []

    # We use some arbitrary seeds to create domains with banjori
    banjori_seeds = ['somestring', 'firetruck', 'bulldozer', 'airplane', 'racecar',
                     'apartment', 'laptop', 'laptopcomp', 'malwareisbad', 'crazytrain',
                     'thepolice', 'fivemonkeys', 'hockey', 'football', 'baseball',
                     'basketball', 'trackandfield', 'fieldhockey', 'softball', 'redferrari',
                     'blackcheverolet', 'yellowelcamino', 'blueporsche', 'redfordf150',
                     'purplebmw330i', 'subarulegacy', 'hondacivic', 'toyotaprius',
                     'sidewalk', 'pavement', 'stopsign', 'trafficlight', 'turnlane',
                     'passinglane', 'trafficjam', 'airport', 'runway', 'baggageclaim',
                     'passengerjet', 'delta1008', 'american765', 'united8765', 'southwest3456',
                     'albuquerque', 'sanfrancisco', 'sandiego', 'losangeles', 'newyork',
                     'atlanta', 'portland', 'seattle', 'washingtondc']

    segs_size = int(max(1, num_per_dga/len(banjori_seeds)))
    for banjori_seed in banjori_seeds:
        domains += banjori.generate_domains(segs_size, banjori_seed)
        labels += ['banjori']*segs_size

    domains += corebot.generate_domains(num_per_dga)
    labels += ['corebot']*num_per_dga

    # Create different length domains using cryptolocker
    crypto_lengths = range(8, 32)
    segs_size = int(max(1, num_per_dga/len(crypto_lengths)))
    for crypto_length in crypto_lengths:
        domains += cryptolocker.generate_domains(segs_size,
                                                 seed_num=random.randint(1, 1000000),
                                                 length=crypto_length)
        labels += ['cryptolocker']*segs_size

    domains += dircrypt.generate_domains(num_per_dga)
    labels += ['dircrypt']*num_per_dga

    # generate kraken and divide between configs
    kraken_to_gen = int(max(1, num_per_dga/2))
    domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'a', 3)
    labels += ['kraken']*kraken_to_gen
    domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'b', 3)
    labels += ['kraken']*kraken_to_gen

    # generate locky and divide between configs
    locky_gen = int(max(1, num_per_dga/11))
    for i in range(1, 12):
        domains += lockyv2.generate_domains(locky_gen, config=i)
        labels += ['locky']*locky_gen

    # Generate pyskpa domains
    domains += pykspa.generate_domains(num_per_dga, datetime(2016, 1, 1))
    labels += ['pykspa']*num_per_dga

    # Generate qakbot
    domains += qakbot.generate_domains(num_per_dga, tlds=[])
    labels += ['qakbot']*num_per_dga

    # ramdo divided over different lengths
    ramdo_lengths = range(8, 32)
    segs_size = int(max(1, num_per_dga/len(ramdo_lengths)))
    for rammdo_length in ramdo_lengths:
        domains += ramdo.generate_domains(segs_size,
                                          seed_num=random.randint(1, 1000000),
                                          length=rammdo_length)
        labels += ['ramdo']*segs_size

    # ramnit
    domains += ramnit.generate_domains(num_per_dga, 0x123abc12)
    labels += ['ramnit']*num_per_dga

    # simda
    simda_lengths = range(8, 32)
    segs_size = int(max(1, num_per_dga/len(simda_lengths)))
    for simda_length in range(len(simda_lengths)):
        domains += simda.generate_domains(segs_size,
                                          length=simda_length,
                                          tld=None,
                                          base=random.randint(2, 2**32))
        labels += ['simda']*segs_size


    return domains, labels

def gen_data(force=False):
    """Grab all data for train/test and save

    force:If true overwrite, else skip if file
          already exists
    """
    if force or (not os.path.isfile(DATA_FILE)):
        if os.path.isfile(IN_HACKATHON_FILE):
            labels = []
            domains = []
            with open(IN_HACKATHON_FILE) as csvfile:
                csvreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
                for row in csvreader:
                    if row['type'] == 'non-generated':
                        labels.append('benign')
                    else:
                        labels.append(row['type'])
                    domains.append(row['domain'])
        elif not os.path.isfile(IN_DATA_FILE):
            domains, labels = gen_malicious(maxSitesNumber)

            # Get equal number of benign/malicious
            lenpre = len(domains)
            domains += get_alexa(len(domains))
            lenpost = len(domains)
            labels += ['benign']*(lenpost-lenpre)

        else:
            labels = []
            domains = []
            with open(IN_DATA_FILE) as csvfile:
                csvreader = csv.DictReader(csvfile, delimiter='|', quotechar='"')
                for row in csvreader:
                    labels.append(row['label'])
                    domains.append(row['domain'])

        zippedlist = list(zip(labels, domains))
        pickle.dump(zippedlist, open(DATA_FILE, 'wb'))

def get_data(force=False):
    """Returns data and labels"""
    gen_data(force)

    return pickle.load(open(DATA_FILE,'rb'))
