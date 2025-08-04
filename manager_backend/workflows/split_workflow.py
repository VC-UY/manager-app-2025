

import uuid

from workflows.models import WorkflowType, Workflow

import os
import pickle
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
    cifar100_dir = os.path.join(dataset_path, "cifar-100-batches-py")
    archive_path = os.path.join(dataset_path, "cifar-100-python.tar.gz")

    if os.path.exists(cifar100_dir):
        return  # Déjà extrait

    if not os.path.exists(archive_path):
        os.makedirs(dataset_path)
        logger.warning(f"⬇️ Téléchargement du dataset CIFAR-100 sur {archive_path}")
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        urllib.request.urlretrieve(url, archive_path)

    logger.warning(f"📦 Extraction du dataset CIFAR-100 sur {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dataset_path)

def generate_openmalaria_scenario(population_size, output_dir, shard_id):
    """
    Génère un fichier de scénario XML pour OpenMalaria avec une population donnée.
    
    Args:
        population_size (int): Taille de la population pour la simulation.
        output_dir (str): Répertoire où sauvegarder le fichier scénario.
        shard_id (int): Identifiant du shard pour nommer le fichier.
    
    Returns:
        str: Chemin du fichier scénario généré.
    """
    # Modèle de scénario XML de base (simplifié, à adapter selon vos besoins)
    scenario_template = """<?xml version='1.0' encoding='UTF-8'?>
<om:scenario xmlns:om="http://openmalaria.org/schema/scenario_47" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" analysisNo="49" name="Idete Incidence{shard_id}" schemaVersion="47" wuID="536305339" xsi:schemaLocation="http://openmalaria.org/schema/scenario_47 scenario_current.xsd">
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
    <parameters interval="5" iseed="0" latentp="3">
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
    scenario_content = scenario_template.format(shard_id=shard_id, population_size=population_size)
    
    os.makedirs(output_dir, exist_ok=True)
    scenario_path = os.path.join(output_dir, f"scenario.xml")
    
    with open(scenario_path, "w") as f:
        f.write(scenario_content)
    
    logger.info(f"Scénario généré pour shard {shard_id} à {scenario_path}")
    return scenario_path

def split_ml_training_workflow(workflow_instance: Workflow, logger:logging.Logger):
    """
    Effectue le découpage pour un workflow ML_TRAINING à partir du script externe.
    """
    dataset_path = os.path.join(workflow_instance.executable_path, "data")
    input_dir = os.path.join(workflow_instance.executable_path, "inputs")

    # S'assurer que le dataset est présent
    download_cifar100_if_needed(dataset_path)

    # Étape 1: déterminer ressources min
    min_resources = get_min_volunteer_resources()

    # Étape 2: estimer nb de shards à partir du dataset complet
    data_batch_path = os.path.join(dataset_path, "cifar-100-python", "train")
    with open(data_batch_path, "rb") as f:
        dataset = pickle.load(f, encoding='latin1')
    dataset_len = len(dataset["data"])

    num_shards = estimate_required_shards(dataset_len, min_resources["min_ram"])
    
    # Étape 3: appeler le script de découpage
    from workflows.examples.cifar100_training.split_dataset import split_dataset
    logger.warning(f"Appel de la fonction de decouppage de ml. Dataset path: {dataset_path}, Output path: {input_dir}")
    # Utiliser le chemin du dataset pour l'entrée et le chemin de base pour la sortie
    split_dataset(num_shards, path=input_dir, dataset_path=dataset_path, logger=logger)
    logger.warning(f"Decouppage en {num_shards} shards Ok.")
    logger.warning("Creation de taches.")

    # Étape 4: création des tâches pour chaque shard
    # docker_img = {
    #     "name": "traning-test",
    #     "tag": "latest"
    # }

    docker_img = {
        "name": "cirfar100-train",
        "tag": "latest"
    }
    tasks = []
    for i in range(num_shards):
        input_size = os.path.getsize(os.path.join(input_dir, f"shard_{i}/data.pkl")) // (1024 * 1024 )  # Convertir en Mo
        # if (input_size > (min_resources["disk"] * 1024)):  # Convertir Go en Mo
        #     logger.error(f"Shard {i} exceeds the minimum disk requirement.")   # Convertir Go en Mo
        #     continue

        # Créer la tâche pour chaque shard
        task = Task.objects.create(
            workflow=workflow_instance,
            name=f"Train Shard {i}",
            description=f"Training on shard {i}",
            command="python train_on_shard.py",
            parameters=[],
            input_files=[f"shard_{i}/data.pkl"],
            output_files=[f"shard_{i}/output/model.pth", f"shard_{i}/output/metrics.json"],
            status= TaskStatus.CREATED,
            parent_task=None,
            is_subtask=False,
            progress=0,
            start_time=None,
            docker_info=docker_img,
            required_resources={
                "cpu": min_resources["min_cpu"],
                "ram": min_resources["min_ram"],
                "disk": min_resources["disk"],
            },
            estimated_max_time=300,
        )
        task.input_size = input_size
        task.save()
        tasks.append(task)
        logger.warning(f"Tache {i}: {task} créée avec succès")
    
    # Étape 5: sauvegarder les tâches dans le workflow
    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks


def split_openmalaria_workflow(workflow_instance: Workflow, num_tasks: int, population_per_task: int, logger: logging.Logger):
    """
    Découpe un workflow OpenMalaria en tâches avec des scénarios distincts.
    
    Args:
        workflow_instance (Workflow): Instance du workflow à découper.
        num_tasks (int): Nombre de tâches à créer.
        population_per_task (int): Taille de la population par tâche.
        logger (logging.Logger): Logger pour les messages.
    
    Returns:
        list: Liste des tâches créées.
    """
    input_dir = os.path.join(workflow_instance.executable_path, "inputs")
    min_resources = get_min_volunteer_resources()
    
    docker_img = {
        "name": "malaria-exp",
        "tag": "latest"
    }
    
    tasks = []
    for i in range(num_tasks):
        # Générer le fichier de scénario
        scenario_path = generate_openmalaria_scenario(
            population_size=population_per_task,
            output_dir=os.path.join(input_dir, f"shard_{i}"),
            shard_id=i
        )
        
        # Calculer la taille du fichier d'entrée
        input_size = os.path.getsize(scenario_path) // (1024 * 1024)  # Convertir en Mo
        
        # Créer la tâche
        task = Task.objects.create(
            workflow=workflow_instance,
            name=f"OpenMalaria Shard {i}",
            description=f"Simulation OpenMalaria sur population {population_per_task}",
            command="/openmalaria/build/openMalaria -s /input/scenario.xml  -o /output/output.txt",
            parameters=[],
            input_files=[f"shard_{i}/scenario.xml"],
            output_files=[f"shard_{i}/output/output.txt"],
            status=TaskStatus.CREATED,
            parent_task=None,
            is_subtask=False,
            progress=0,
            start_time=None,
            docker_info=docker_img,
            required_resources={
                "cpu": min_resources["min_cpu"],
                "ram": min_resources["min_ram"],
                "disk": max(min_resources["disk"], input_size / 1024 + 1),  # Ajouter 1 Go pour les sorties
            },
            estimated_max_time=200,  # 5 min
        )
        task.input_size = input_size
        task.save()
        tasks.append(task)
        logger.warning(f"Tâche {i}: {task} créée avec succès")
    
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
    else:
        raise ValueError(f"Type de workflow non supporté: {workflow_type}")
    
    return tasks
