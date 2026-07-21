

import uuid

from workflows.models import WorkflowType, Workflow

import json
import os
import pickle
import shutil
from pathlib import Path
from tasks.models import Task, TaskStatus
from volunteers.models import Volunteer
from workflows.models import WorkflowType
import logging
import tarfile
import urllib.request
import math
from django.conf import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
manager_host = settings.MANAGER_HOST

def get_min_volunteer_resources():
    """Retourne les ressources du volontaire le plus faible (RAM, CPU)."""
    volunteers = Volunteer.objects.all()
    if not volunteers:
        return {
            "min_cpu": 1,
            "min_ram": 512,
            "disk": 1, # en Go
        }
    return {
        "min_cpu": min(v.cpu_cores for v in volunteers),
        "min_ram": min(v.ram_mb for v in volunteers),
        "disk": min(v.disk_gb for v in volunteers),
    }


def estimate_required_shards(dataset_len, min_ram_mb):
    """
    Estime le nombre de shards à créer pour que chaque shard tienne dans la mémoire minimale disponible.
    
    Chaque shard aura autant d'échantillons que possible sans dépasser min_ram_mb.
    """
    # Estimation : chaque échantillon ~0.07MB (32x32x3 uint8 ≈ 3KB, soit ~0.003MB + métadonnées + batch + surcharge)
    est_sample_size_mb = 0.07  

    max_samples_per_shard = int(min_ram_mb / est_sample_size_mb)
    if max_samples_per_shard < 1:
        max_samples_per_shard = 1  # éviter division par 0

    num_shards = math.ceil(dataset_len / max_samples_per_shard)

    return max(1, num_shards)

def download_cifar10_if_needed(dataset_path):
    cifar10_dir = os.path.join(dataset_path, "cifar-10-batches-py")
    archive_path = os.path.join(dataset_path, "cifar-10-python.tar.gz")

    if os.path.exists(cifar10_dir):
        return  # Déjà extrait

    if not os.path.exists(archive_path):
        logger.warning(f"⬇️ Téléchargement du dataset CIFAR-10 sur {archive_path}")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        urllib.request.urlretrieve(url, archive_path)

    logger.warning(f"📦 Extraction du dataset CIFAR-10 sur {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dataset_path)

    

def download_cifar100_if_needed(dataset_path):
    cifar100_dir = os.path.join(dataset_path, "cifar-100-python")
    archive_path = os.path.join(dataset_path, "cifar-100-python.tar.gz")

    if os.path.exists(os.path.join(cifar100_dir, "train")):
        return  # Déjà extrait

    os.makedirs(dataset_path, exist_ok=True)
    if not os.path.exists(archive_path):
        logger.warning(f"Telechargement du dataset CIFAR-100 sur {archive_path}")
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        urllib.request.urlretrieve(url, archive_path)

    logger.warning(f"Extraction du dataset CIFAR-100 sur {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dataset_path)

def generate_openmalaria_scenario(
    population_size,
    output_dir,
    shard_id,
    study_name="GlobalMalariaStudy",
    seed=0,
):
    """
    Génère le scénario XML d'une PARTITION d'une étude globale.

    Tous les shards partagent la même étude (paramètres épidémiologiques identiques).
    Seuls popSize et iseed varient — c'est le partitionnement d'une population globale.
    """
    scenario_template = """<?xml version='1.0' encoding='UTF-8'?>
<om:scenario xmlns:om="http://openmalaria.org/schema/scenario_47" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" analysisNo="49" name="{study_name}_partition_{shard_id}" schemaVersion="47" wuID="536305339" xsi:schemaLocation="http://openmalaria.org/schema/scenario_47 scenario_current.xsd">
  <demography name="Ifakara" maximumAgeYrs="100" popSize="{population_size}">
    <ageGroup lowerbound="0.0">
      <group poppercent="3.474714994" upperbound="1"/>
      <group poppercent="12.76004028" upperbound="5"/>
      <group poppercent="14.52151394" upperbound="10"/>
      <group poppercent="12.75565434" upperbound="15"/>
      <group poppercent="10.83632374" upperbound="20"/>
      <group poppercent="8.393312454" upperbound="25"/>
      <group poppercent="7.001421452" upperbound="30"/>
      <group poppercent="5.800587654" upperbound="35"/>
      <group poppercent="5.102136612" upperbound="40"/>
      <group poppercent="4.182561874" upperbound="45"/>
      <group poppercent="3.339409351" upperbound="50"/>
      <group poppercent="2.986112356" upperbound="55"/>
      <group poppercent="2.555766582" upperbound="60"/>
      <group poppercent="2.332763433" upperbound="65"/>
      <group poppercent="1.77400255" upperbound="70"/>
      <group poppercent="1.008525491" upperbound="75"/>
      <group poppercent="0.74167341" upperbound="80"/>
      <group poppercent="0.271863401" upperbound="85"/>
      <group poppercent="0.161614642" upperbound="90"/>
    </ageGroup>
  </demography>
  <monitoring name="Idete">
    <SurveyOptions>
      <option name="nHost" value="true"/>
      <option name="nUncomp" value="true"/>
      <option name="sumAge" value="true"/>
    </SurveyOptions>
    <surveys diagnostic="standard">
      <surveyTime>1y</surveyTime>
    </surveys>
    <ageGroup lowerbound="0.0">
      <group upperbound="0.25"/>
      <group upperbound="0.5"/>
      <group upperbound="0.75"/>
      <group upperbound="1.0"/>
    </ageGroup>
  </monitoring>
  <interventions name="No Intervention"/>
  <healthSystem>
    <ImmediateOutcomes name="Ironmal">
      <drugRegimen firstLine="CQ" inpatient="CQ" secondLine="CQ"/>
      <initialACR>
        <CQ value="0.6"/>
        <SP value="0"/>
        <AQ value="0"/>
        <ACT value="0"/>
        <QN value="0"/>
        <selfTreatment value="0"/>
      </initialACR>
      <compliance>
        <CQ value="1"/>
        <SP value="0"/>
        <AQ value="0"/>
        <ACT value="0"/>
        <QN value="0"/>
        <selfTreatment value="0"/>
      </compliance>
      <nonCompliersEffective>
        <CQ value="0"/>
        <SP value="0"/>
        <AQ value="0"/>
        <ACT value="0"/>
        <QN value="0"/>
        <selfTreatment value="0"/>
      </nonCompliersEffective>
      <treatmentActions>
        <CQ name="clear blood-stage infections">
          <clearInfections stage="blood" timesteps="1"/>
        </CQ>
      </treatmentActions>
      <pSeekOfficialCareUncomplicated1 value="0.64"/>
      <pSelfTreatUncomplicated value="0"/>
      <pSeekOfficialCareUncomplicated2 value="0.64"/>
      <pSeekOfficialCareSevere value="0.48"/>
    </ImmediateOutcomes>
    <CFR>
      <group lowerbound="0" value="0.09189"/>
      <group lowerbound="0.25" value="0.0810811"/>
      <group lowerbound="0.75" value="0.0648649"/>
      <group lowerbound="1.5" value="0.0689189"/>
      <group lowerbound="2.5" value="0.0675676"/>
      <group lowerbound="3.5" value="0.0297297"/>
      <group lowerbound="4.5" value="0.0459459"/>
      <group lowerbound="7.5" value="0.0945946"/>
      <group lowerbound="12.5" value="0.1243243"/>
      <group lowerbound="15" value="0.1378378"/>
    </CFR>
    <pSequelaeInpatient interpolation="none">
      <group lowerbound="0.0" value="0.0132"/>
      <group lowerbound="5.0" value="0.005"/>
    </pSequelaeInpatient>
  </healthSystem>
  <entomology mode="forced" name="Idete">
    <!--first day is 17-03-92-->
    <nonVector eipDuration="10">
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>31.5389</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>12.0794</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>30.3456</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4530</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>0.4174</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>1.0581</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.7063</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.4828</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>0.2606</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>1.7687</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.3815</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.0629</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1237</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.1351</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.0829</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.1222</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0547</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.0196</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.1861</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.3604</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.2309</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3349</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.3663</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.2367</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.5726</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1705</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.1684</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.0905</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.6006</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>0.1915</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>1.1981</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.9569</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.4506</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1157</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.1217</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.0499</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.1119</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.9067</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.4750</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.6395</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.2549</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.3137</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.0876</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.2192</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.3620</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4264</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4314</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.4191</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.2827</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.4949</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.6872</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>0.4083</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>4.0790</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.3345</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.4434</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.7719</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.4817</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>0.3894</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>1.2806</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.2673</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>0.4734</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
      <EIRDaily>1.8254</EIRDaily>
    </nonVector>
  </entomology>
  <diagnostics>
    <diagnostic name="standard" units="Other">
      <deterministic minDensity="40"/>
    </diagnostic>
    <diagnostic name="neonatal" units="Other">
      <deterministic minDensity="40"/>
    </diagnostic>
  </diagnostics>
  <model>
    <ModelOptions>
      <option name="MUELLER_PRESENTATION_MODEL" value="true"/>
      <option name="MAX_DENS_CORRECTION" value="false"/>
      <option name="INNATE_MAX_DENS" value="false"/>
      <option name="INDIRECT_MORTALITY_FIX" value="false"/>
    </ModelOptions>
    <clinical healthSystemMemory="6">
      <NeonatalMortality diagnostic="neonatal"/>
    </clinical>
    <human>
      <availabilityToMosquitoes>
        <group lowerbound="0.0" value="0.225940909648"/>
        <group lowerbound="1.0" value="0.286173633441"/>
        <group lowerbound="2.0" value="0.336898395722"/>
        <group lowerbound="3.0" value="0.370989854675"/>
        <group lowerbound="4.0" value="0.403114915112"/>
        <group lowerbound="5.0" value="0.442585112522"/>
        <group lowerbound="6.0" value="0.473839351511"/>
        <group lowerbound="7.0" value="0.512630464378"/>
        <group lowerbound="8.0" value="0.54487872702"/>
        <group lowerbound="9.0" value="0.581527755812"/>
        <group lowerbound="10.0" value="0.630257580698"/>
        <group lowerbound="11.0" value="0.663063362714"/>
        <group lowerbound="12.0" value="0.702417432755"/>
        <group lowerbound="13.0" value="0.734605377277"/>
        <group lowerbound="14.0" value="0.788908765653"/>
        <group lowerbound="15.0" value="0.839587932303"/>
        <group lowerbound="20.0" value="1.0"/>
        <group lowerbound="20.0" value="1.0"/>
      </availabilityToMosquitoes>
    </human>
    <parameters interval="5" iseed="{seed}" latentp="3">
      <parameter name="         '-ln(1-Sinf)'    " number="1" value="0.050736"/>
      <parameter name="         Estar    " number="2" value="0.03247"/>
      <parameter name="         Simm     " number="3" value="0.157325"/>
      <parameter name="         Xstar_p  " number="4" value="2393.949859"/>
      <parameter name="         gamma_p  " number="5" value="1.979441"/>
      <parameter name="         sigma2i  " number="6" value="9.525457"/>
      <parameter name="         CumulativeYstar  " number="7" value="151465400.748812"/>
      <parameter name="         CumulativeHstar  " number="8" value="70.526914"/>
      <parameter name="         '-ln(1-alpha_m)'         " number="9" value="2.349838"/>
      <parameter name="         decay_m  " number="10" value="2.372811"/>
      <parameter name="         sigma2_0         " number="11" value="0.657622"/>
      <parameter name="         Xstar_v  " number="12" value="0.922477"/>
      <parameter name="         Ystar2   " number="13" value="10004.145044"/>
      <parameter name="         alpha    " number="14" value="141306.48626"/>
      <parameter name="         Density bias (non Garki)         " number="15" value="0.156321"/>
      <parameter name="         No Use 1         " number="16" value="1"/>
      <parameter name="         log oddsr CF community   " number="17" value="0.712956"/>
      <parameter name="         Indirect risk cofactor   " number="18" value="0.013118"/>
      <parameter name="         Non-malaria infant mortality     " number="19" value="60.798982"/>
      <parameter name="         Density bias (Garki)     " number="20" value="5.561993"/>
      <parameter name="         Severe Malaria Threshhold        " number="21" value="374899.564569"/>
      <parameter name="         Immunity Penalty         " number="22" value="1"/>
      <parameter name=" Immune effector decay " number="23" value="0"/>
      <parameter name="         comorbidity intercept    " number="24" value="0.091105"/>
      <parameter name="         Ystar half life  " number="25" value="0.281908"/>
      <parameter name="         Ystar1   " number="26" value="0.602292"/>
      <parameter name=" Asex immune decay " number="27" value="9.5e-05"/>
      <parameter name="         Ystar0   " number="28" value="541.4835"/>
      <parameter name="         Idete multiplier         " number="29" value="2.83077"/>
      <parameter name="         critical age for comorbidity     " number="30" value="0.105099"/>
      <parameter name="Mueller dummy 1" number="31" value="2.797523626"/>
      <parameter name="Mueller dummy 2" number="32" value="0.117383"/>
    </parameters>
  </model>
</om:scenario>
"""
    scenario_content = scenario_template.format(
        shard_id=shard_id,
        population_size=population_size,
        study_name=study_name,
        seed=int(seed),
    )

    os.makedirs(output_dir, exist_ok=True)
    scenario_path = os.path.join(output_dir, "scenario.xml")

    with open(scenario_path, "w", encoding="utf-8") as f:
        f.write(scenario_content)

    logger.info("Partition %s: scénario écrit dans %s (pop=%s)", shard_id, scenario_path, population_size)
    return scenario_path

def _write_partitioned_ml_dataset(input_dir, num_shards, samples_per_shard, epochs, logger):
    """
    Construit UN jeu de données global, puis le partitionne en shards contigus.

    Manifeste global: inputs/global_dataset_manifest.json
    Chaque shard: data.pkl + partition.json (offset global, taille, epochs)
    """
    import random

    os.makedirs(input_dir, exist_ok=True)
    samples_per_shard = max(1000, int(samples_per_shard))
    total_samples = num_shards * samples_per_shard
    study_id = f"ml-study-{total_samples}"

    manifest = {
        "study_id": study_id,
        "paradigm": "partition_train_aggregate",
        "description": (
            "Jeu de données global partitionné en shards disjoints. "
            "Chaque volontaire entraîne un modèle local; le Manager agrège "
            "les poids (moyenne fédérée) pour reconstruire le modèle global."
        ),
        "total_samples": total_samples,
        "num_partitions": num_shards,
        "samples_per_partition": samples_per_shard,
        "epochs": epochs,
        "num_classes": 100,
        "input_shape": [32, 32, 3],
        "partitions": [],
    }

    rng = random.Random(42)
    for i in range(num_shards):
        offset = i * samples_per_shard
        shard_dir = os.path.join(input_dir, f"shard_{i}")
        os.makedirs(shard_dir, exist_ok=True)
        # Données déterministes par index global (partition réelle d'un corpus unique)
        data = [
            [
                [[((offset + s) * 3 + c + r + col) % 256 for c in range(3)] for col in range(32)]
                for r in range(32)
            ]
            for s in range(samples_per_shard)
        ]
        labels = [((offset + s) * 7 + rng.randint(0, 3)) % 100 for s in range(samples_per_shard)]
        with open(os.path.join(shard_dir, "data.pkl"), "wb") as handle:
            pickle.dump((data, labels), handle)

        partition = {
            "study_id": study_id,
            "partition_index": i,
            "global_offset": offset,
            "num_samples": samples_per_shard,
            "total_samples": total_samples,
            "epochs": epochs,
            "num_classes": 100,
        }
        with open(os.path.join(shard_dir, "partition.json"), "w", encoding="utf-8") as handle:
            json.dump(partition, handle, indent=2)

        manifest["partitions"].append(
            {
                "partition_index": i,
                "global_offset": offset,
                "num_samples": samples_per_shard,
                "path": f"shard_{i}",
            }
        )
        if logger:
            logger.warning(
                "Partition ML %s/%s: offset=%s samples=%s",
                i + 1,
                num_shards,
                offset,
                samples_per_shard,
            )

    with open(os.path.join(input_dir, "global_dataset_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def split_ml_training_workflow(workflow_instance: Workflow, logger: logging.Logger):
    """
    Partitionne un jeu de données global puis crée une tâche d'entraînement par partition.
    Agrégation ultérieure = fusion des modèles locaux (federated averaging).
    """
    dataset_path = os.path.join(workflow_instance.executable_path, "data")
    input_dir = os.path.join(workflow_instance.executable_path, "inputs")
    min_resources = get_min_volunteer_resources()

    metadata = workflow_instance.metadata or {}
    # Charge réaliste: au moins 7–8 partitions
    num_shards = max(7, int(metadata.get("num_tasks") or 8))
    samples_per_shard = int(metadata.get("samples_per_shard") or 6000)
    epochs = int(metadata.get("epochs") or 25)
    use_synthetic = metadata.get("synthetic", True)

    os.makedirs(input_dir, exist_ok=True)

    if use_synthetic:
        logger.warning(
            "Partitionnement ML global: %s partitions × %s samples, %s epochs",
            num_shards,
            samples_per_shard,
            epochs,
        )
        manifest = _write_partitioned_ml_dataset(
            input_dir, num_shards, samples_per_shard, epochs, logger
        )
    else:
        download_cifar100_if_needed(dataset_path)
        from workflows.examples.cifar100_training.split_dataset import split_dataset

        split_dataset(
            num_shards,
            path=input_dir,
            dataset_path=dataset_path,
            logger=logger,
            samples_per_shard=samples_per_shard,
        )
        manifest = {
            "study_id": f"cifar100-{num_shards}",
            "paradigm": "partition_train_aggregate",
            "total_samples": num_shards * samples_per_shard,
            "num_partitions": num_shards,
            "epochs": epochs,
        }
        with open(os.path.join(input_dir, "global_dataset_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    metadata.update(
        {
            "num_tasks": num_shards,
            "samples_per_shard": samples_per_shard,
            "epochs": epochs,
            "total_samples": manifest.get("total_samples"),
            "paradigm": "partition_train_aggregate",
        }
    )
    workflow_instance.metadata = metadata
    workflow_instance.save(update_fields=["metadata", "updated_at"])

    from workflows.bundle_builder import RUNTIME_META, package_files_as_bundle

    docker_img = dict(RUNTIME_META)
    worker_script = Path(__file__).resolve().parent / "examples" / "ml_training" / "train_on_shard.py"
    tasks = []
    for i in range(num_shards):
        shard_dir = os.path.join(input_dir, f"shard_{i}")
        data_path = os.path.join(shard_dir, "data.pkl")
        partition_path = os.path.join(shard_dir, "partition.json")
        files = [data_path]
        if os.path.isfile(partition_path):
            files.append(partition_path)
        input_size = max(1, os.path.getsize(data_path) // (1024 * 1024))

        bundle_name = "task_bundle.tar.gz"
        bundle_path = os.path.join(shard_dir, bundle_name)
        package_files_as_bundle(
            files=files,
            command="python3 train_on_shard.py",
            bundle_path=bundle_path,
            worker_scripts=[worker_script] if worker_script.is_file() else None,
        )
        input_size = max(input_size, max(1, os.path.getsize(bundle_path) // (1024 * 1024)))

        task = Task.objects.create(
            workflow=workflow_instance,
            name=f"Train Partition {i}",
            description=(
                f"Entraînement local sur partition {i}/{num_shards} "
                f"du jeu global ({samples_per_shard} samples, {epochs} epochs)"
            ),
            command="python3 train_on_shard.py",
            parameters=[],
            input_files=[f"shard_{i}/{bundle_name}"],
            output_files=["model.pt", "metrics.json"],
            status=TaskStatus.CREATED,
            parent_task=None,
            is_subtask=False,
            progress=0,
            start_time=None,
            docker_info=docker_img,
            required_resources={
                "cpu": min_resources["min_cpu"],
                "ram": max(min_resources["min_ram"], 2048),
                "disk": min_resources["disk"],
            },
            estimated_max_time=max(1800, epochs * 60),
        )
        task.input_size = input_size
        task.save()
        tasks.append(task)
        logger.warning("Tâche ML partition %s créée (bundle): %s", i, task.id)

    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks


def split_openmalaria_workflow(
    workflow_instance: Workflow,
    num_tasks: int,
    population_per_task: int,
    logger: logging.Logger,
):
    """
    Étude épidémiologique GLOBALE partitionnée en sous-populations.

    1. Définit une étude globale (population totale, durée, paramètres partagés)
    2. Partitionne la population en sous-ensembles contigus (une tâche = une partition)
    3. Chaque volontaire simule sa sous-population
    4. Le Manager agrège les sorties en indicateurs globaux pondérés
    """
    metadata = workflow_instance.metadata or {}
    num_tasks = max(7, int(num_tasks or metadata.get("num_tasks") or 8))
    total_population = int(
        metadata.get("total_population")
        or (num_tasks * int(population_per_task or 20000))
    )
    # Recalcule des tailles de partition (dernière partition absorbe le reste)
    base_pop = total_population // num_tasks
    simulation_days = int(metadata.get("simulation_days") or 3650)
    monte_carlo_runs = int(metadata.get("monte_carlo_runs") or 12)
    study_name = metadata.get("study_name") or f"MalariaStudy_{workflow_instance.id}"
    study_id = str(workflow_instance.id)

    epidemiology = {
        "bite_rate": float(metadata.get("bite_rate", 0.3)),
        "transmission_mh": float(metadata.get("transmission_mh", 0.5)),
        "recovery_rate": float(metadata.get("recovery_rate", 0.05)),
        "mosquito_density": float(metadata.get("mosquito_density", 2.0)),
        "max_agents": int(metadata.get("max_agents", 15000)),
    }

    input_dir = os.path.join(workflow_instance.executable_path, "inputs")
    os.makedirs(input_dir, exist_ok=True)
    min_resources = get_min_volunteer_resources()
    # Plus d'image Docker : exécution via bundle self-contained vc-uyr
    docker_img = {
        "runtime": "vc-uyr",
        "bundle": True,
    }

    worker_script = Path(__file__).resolve().parent / "examples" / "openmalaria_worker" / "run_simulation.py"
    if not worker_script.is_file():
        raise FileNotFoundError(f"Worker malaria introuvable: {worker_script}")

    from workflows.bundle_builder import create_task_bundle


    global_study = {
        "study_id": study_id,
        "study_name": study_name,
        "paradigm": "partition_simulate_aggregate",
        "description": (
            "Étude épidémiologique globale sur une population totale. "
            "La population est partitionnée en sous-populations disjointes; "
            "chaque volontaire exécute la simulation sur sa partition; "
            "les résultats sont agrégés (prévalence pondérée, cas totaux, EIR)."
        ),
        "total_population": total_population,
        "num_partitions": num_tasks,
        "simulation_days": simulation_days,
        "monte_carlo_runs": monte_carlo_runs,
        "epidemiology": epidemiology,
        "partitions": [],
    }

    tasks = []
    offset = 0
    for i in range(num_tasks):
        # Dernière partition prend le reste pour couvrir toute la population
        if i == num_tasks - 1:
            pop_i = total_population - offset
        else:
            pop_i = base_pop
        shard_dir = os.path.join(input_dir, f"shard_{i}")
        os.makedirs(shard_dir, exist_ok=True)

        scenario_path = generate_openmalaria_scenario(
            population_size=pop_i,
            output_dir=shard_dir,
            shard_id=i,
            study_name=study_name,
            seed=1000 + i,
        )

        partition = {
            "study_id": study_id,
            "study_name": study_name,
            "partition_index": i,
            "num_partitions": num_tasks,
            "population_offset": offset,
            "population_size": pop_i,
            "total_population": total_population,
            "simulation_days": simulation_days,
            "monte_carlo_runs": monte_carlo_runs,
            "seed": 1000 + i,
            "epidemiology": epidemiology,
        }
        partition_path = os.path.join(shard_dir, "partition.json")
        with open(partition_path, "w", encoding="utf-8") as handle:
            json.dump(partition, handle, indent=2)

        global_study["partitions"].append(
            {
                "partition_index": i,
                "population_offset": offset,
                "population_size": pop_i,
                "path": f"shard_{i}",
            }
        )

        input_size = max(1, os.path.getsize(scenario_path) // (1024 * 1024))

        # Bundle self-contained : run.sh + run_simulation.py + entrées du shard
        staging_dir = os.path.join(shard_dir, "bundle_staging")
        os.makedirs(staging_dir, exist_ok=True)
        shutil.copy2(worker_script, os.path.join(staging_dir, "run_simulation.py"))
        shutil.copy2(scenario_path, os.path.join(staging_dir, "scenario.xml"))
        shutil.copy2(partition_path, os.path.join(staging_dir, "partition.json"))
        bundle_name = f"shard_{i}_bundle.tar.gz"
        bundle_path = os.path.join(shard_dir, bundle_name)
        create_task_bundle(
            staging_dir=staging_dir,
            bundle_path=bundle_path,
            command="python3 run_simulation.py",
        )
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            pass
        input_size = max(input_size, max(1, os.path.getsize(bundle_path) // (1024 * 1024)))

        # Estimation par partition en mode test: 2 à 5 minutes.
        est_seconds = max(
            120,
            min(
                300,
                int(
                    (min(pop_i, epidemiology["max_agents"]) * simulation_days * monte_carlo_runs)
                    / 500_000
                ),
            ),
        )

        task = Task.objects.create(
            workflow=workflow_instance,
            name=f"Malaria Partition {i}",
            description=(
                f"Partition {i}/{num_tasks} de l'étude globale "
                f"(individus {offset}–{offset + pop_i - 1}, "
                f"{simulation_days} jours, {monte_carlo_runs} réplicats MC)"
            ),
            command="python3 run_simulation.py",
            parameters=[],
            input_files=[f"shard_{i}/{bundle_name}"],
            output_files=["output.txt", "partition_metrics.json"],
            status=TaskStatus.CREATED,
            parent_task=None,
            is_subtask=False,
            progress=0,
            start_time=None,
            docker_info=docker_img,
            required_resources={
                "cpu": min_resources["min_cpu"],
                "ram": max(min_resources["min_ram"], 2048),
                "disk": max(min_resources["disk"], 2),
            },
            estimated_max_time=est_seconds,
        )
        task.input_size = input_size
        task.save()
        tasks.append(task)
        logger.warning(
            "Partition malaria %s: pop=%s offset=%s task=%s est=%ss",
            i,
            pop_i,
            offset,
            task.id,
            est_seconds,
        )
        offset += pop_i

    with open(os.path.join(input_dir, "global_study.json"), "w", encoding="utf-8") as handle:
        json.dump(global_study, handle, indent=2)

    metadata.update(
        {
            "num_tasks": num_tasks,
            "total_population": total_population,
            "population_per_task": base_pop,
            "simulation_days": simulation_days,
            "monte_carlo_runs": monte_carlo_runs,
            "paradigm": "partition_simulate_aggregate",
            "study_name": study_name,
        }
    )
    workflow_instance.metadata = metadata
    workflow_instance.save(update_fields=["metadata", "updated_at"])

    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks


def split_workflow(id: uuid.UUID, workflow_type: WorkflowType, logger, num_tasks: int = None, population_per_task: int = None):
    """
    Découpe un workflow en tâches plus petites selon le type de workflow.
    
    Args:
        id (uuid.UUID): ID du workflow à découper.
        workflow_type (WorkflowType): Type du workflow.
        logger: Logger pour les messages.
        num_tasks (int, optional): Nombre de tâches pour OpenMalaria.
        population_per_task (int, optional): Taille de la population par tâche pour OpenMalaria.
    
    Returns:
        list: Liste des tâches créées.
    """
    workflow_instance = Workflow.objects.get(id=id)
    
    if workflow_type == WorkflowType.ML_TRAINING:
        tasks = split_ml_training_workflow(workflow_instance, logger)
    elif workflow_type == WorkflowType.OPEN_MALARIA:
        if num_tasks is None or population_per_task is None:
            raise ValueError("num_tasks et population_per_task doivent être spécifiés pour OpenMalaria")
        tasks = split_openmalaria_workflow(workflow_instance, num_tasks, population_per_task, logger)
    elif workflow_type == WorkflowType.MATRIX_ADDITION:
        from workflows.split_generic import split_matrix_workflow
        n = num_tasks or (workflow_instance.metadata or {}).get('num_tasks', 4)
        tasks = split_matrix_workflow(workflow_instance, 'add', logger, num_tasks=int(n))
    elif workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
        from workflows.split_generic import split_matrix_workflow
        n = num_tasks or (workflow_instance.metadata or {}).get('num_tasks', 4)
        tasks = split_matrix_workflow(workflow_instance, 'multiply', logger, num_tasks=int(n))
    elif workflow_type == WorkflowType.ML_INFERENCE:
        from workflows.split_generic import split_ml_inference_workflow
        tasks = split_ml_inference_workflow(workflow_instance, logger)
    elif workflow_type == WorkflowType.CUSTOM:
        from workflows.split_generic import split_custom_workflow
        tasks = split_custom_workflow(workflow_instance, logger)
    elif workflow_type == WorkflowType.DISTRIBUTED_LEARNING:
        from workflows.split_distributed_learning import split_distributed_learning_workflow
        tasks = split_distributed_learning_workflow(workflow_instance, logger)
    else:
        raise ValueError(f"Type de workflow non supporté: {workflow_type}")
    
    return tasks
