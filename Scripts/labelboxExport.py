import labelbox as lb
import json
import time
import os
import urllib.request 
from PIL import Image , ImageOps

# Source: https://github.com/Labelbox/labelbox-python/blob/master/examples/basics/data_rows.ipynb

### From LabelBox ###

refreshJson = False

if refreshJson:
  LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHQxcno4cHkxNjhtMDcwYzhxNHliaGRwIiwib3JnYW5pemF0aW9uSWQiOiJjbHQxcno4cG8xNjhsMDcwYzZnb3I4NHY3IiwiYXBpS2V5SWQiOiJjbHQzZzVleWUwNWU4MDd0c2Y0NHgzZnc4Iiwic2VjcmV0IjoiNzJiOGJmNWU4ZGY4ODQ2MjM1NWQ0MjlmMTlkZGNhNDQiLCJpYXQiOjE3MDg5ODI2NTgsImV4cCI6MjM0MDEzNDY1OH0.KZenirjxyWNFXLeSbMtsPKs1ImDjCZKzp_QCi1tZ4Do'
  PROJECT_ID = 'clt3bojz4029807tsgec6bclb'
  client = lb.Client(api_key = LB_API_KEY, enable_experimental=True)
  project = client.get_project(PROJECT_ID)
  labels = project.export_v2(params={
    "data_row_details": True,
    "metadata_fields": True,
    "attachments": True,
    "project_details": True,
    "performance_details": True,
    "label_details": True,
    "interpolated_frames": True
    })


  batches = list(project.batches())
  batch_ids = [batch.uid for batch in batches]


  export_params = {
  "attachments": True,
    "metadata_fields": True,
    "data_row_details": True,
    "project_details": True,
    "performance_details": True,
    "batch_ids" : batch_ids # Include batch ids if you only want to export specific batches, otherwise,
    #you can export all the data without using this parameter
  }
  filters = {}

  export_task = project.export(params=export_params, filters=filters)
  export_task.wait_till_done()


  data_rows = []

  def json_stream_handler(output: lb.JsonConverterOutput):
    data_row = json.loads(output.json_str)
    data_rows.append(data_row)


  if export_task.has_errors():
    export_task.get_stream(
    
    converter=lb.JsonConverter(),
    stream_type=lb.StreamType.ERRORS
    ).start(stream_handler=lambda error: print(error))

  if export_task.has_result():
    export_json = export_task.get_stream(
      converter=lb.JsonConverter(),
      stream_type=lb.StreamType.RESULT
    ).start(stream_handler=json_stream_handler)

  jsonFile = open(".\Docs\data_new.json", "w")
  jsonFile.write(json.dumps(data_rows))
  jsonFile.close()


def read_emotions(data_rows):
  imported = []
  errors = []
  skipped = []
  happy = 0
  neutral = 0
  focused = 0
  surprised = 0

  target = '.\\Datasets\\Real_and_Fake_Face_Detection\\'
  dir_path = '.'

  pictures_to_import = len(data_rows)
  i = 1
  for data_row in data_rows:
    for id in data_row["projects"]:
      try:
        picture_id = data_row["data_row"]["external_id"]
        time.sleep(0.01)
        print(f'\r{format((i / pictures_to_import),".2%")}', end='\r')
        i += 1
        if data_row["projects"][id]["labels"][0]["performance_details"]["skipped"]:
          skipped.append(picture_id)
        else:
          emotion = data_row["projects"][id]["labels"][0]["annotations"]["classifications"][0]["radio_answer"]["value"]
          match str(emotion).lower():
            case "happy":
              happy += 1
            case "neutral":
              neutral +=1
            case "focused":
              focused += 1
            case "surprised":
              surprised += 1
          imported.append(picture_id)
          if not os.path.isdir(target + '\\' + emotion):
            os.mkdir(target + '\\' + emotion)
          url=data_row["data_row"]['row_data']
          urllib.request.urlretrieve(url, data_row["data_row"]["external_id"])
          img = Image.open(data_row["data_row"]["external_id"])
          img.save(target + "\\" + emotion + '\\' + data_row["data_row"]["external_id"])
          os.remove(dir_path + '\\' + data_row["data_row"]["external_id"])
      except Exception as err:
        print(f'{err=}')
        errors.append(picture_id)
  print()
  print(f'{len(imported)=}')
  print(f'\t{happy=}')
  print(f'\t{neutral=}')
  print(f'\t{focused=}')
  print(f'\t{surprised=}')
  print(f'{len(skipped)=}')
  print(f'{len(errors)=}')



f = open('.\Docs\data_new.json')
data_rows_new = json.load(f)

f.close()

read_emotions(data_rows_new)



