# VLM Exploration 

VLM tester via Gradio webapp

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
