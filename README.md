# VLM Exploration 

VLM tester via Gradio webapp

<img width="1288" height="942" alt="image" src="https://github.com/user-attachments/assets/7288efa9-1362-4fca-8b09-9bc1987018b8" />


## Structure
```
vlm_app_srp/
  app.py
  requirements.txt
  core/
    config.py
    interfaces.py
    postproc.py
    registry.py
    adapters/
      blip2.py
      instructblip.py
      smolvlm.py
      llava.py
      owlvit.py
      tinyclip.py
      groundingdino.py
      etc..
```

## Run
```
pip install -r requirements.txt
python app.py
```
