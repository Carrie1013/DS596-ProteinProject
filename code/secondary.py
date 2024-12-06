import time
import json
import requests
import pandas as pd
from sklearn.utils import resample


# use this function to call psipred api
def call_psipred_api(sequence: str, email):

    psipred = "http://bioinf.cs.ucl.ac.uk/psipred/api"
    submit_url = f"{psipred}/submission"
    fasta_sequence = f">query\n{sequence}"

    payload = {'input_data': fasta_sequence}
    data = {'job': 'psipred', 'submission_name': 'test','email': email}
    r = requests.post(f"{submit_url}.json", data=data, files=payload)
    response_data = json.loads(r.text)
    print(response_data)
    uuid = response_data['UUID']

    retries = 0
    while retries < 30:
      result_uri = f"{submit_url}/{uuid}"
      r = requests.get(result_uri, headers={"Accept":"application/json"})
      result_data = json.loads(r.text)
      if "Complete" in result_data["state"]:
          data_path = result_data['submissions'][0]['results'][5]['data_path']
          response = requests.get(f"{psipred}{data_path}")
          if response.status_code != 200:
              raise Exception(f"Failed to get results: {response.text}")
          ss_sequence = ""
          for line in response.text.splitlines():
              if not line.startswith('#') and len(line.split()) > 2:
                  ss_sequence += line.split()[2]
          return ss_sequence
      else:
          retries += 1
          time.sleep(30)

    raise Exception("Timeout waiting for PSIPRED results")


def proc():
    split_data = pd.read_csv('../data/split.csv')
    split_data = split_data[['Split Site', 'Sequence']]
    split_data['Sequence'] = split_data['Sequence'].str.replace(' ', '', regex=False)

    expanded_rows = split_data['Split Site'].str.split('/').explode()
    expanded_data = pd.DataFrame({
        'Split Site': expanded_rows,
        'Sequence': split_data.loc[expanded_rows.index, 'Sequence'].values
    })
    expanded_data.reset_index(drop=True, inplace=True)

    return expanded_data