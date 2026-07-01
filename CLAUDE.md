A python app to take a folder of image files, create captions using a local VLM, rename the file from caption.

Product Features:

- Find duplicates and move one of them to another folder
- Convert RAW files to jpeg and move RAW file to another folder
- If required, create a resized version suitable for sending to the model
- Get a caption and short title from the model
- Rename the image with the title and create an MD file named with the title containing the caption that has a link to the image
- Save to an output folder
- The MD files should use the OKF standard

To Discuss:

- I'd like to do a pass over the MD files to categorize on organize the output
  - Is another model required for this?
  - Or can it be done in a single pass with the VLM?
  - Or another model running in parallel?
- I have a Strix Halo laptop with 128GB ram
- There are some model options in [compass_artifact_wf-2e149159-73b4-4a03-962a-906ccbb751d6_text_markdown.md](compass_artifact_wf-2e149159-73b4-4a03-962a-906ccbb751d6_text_markdown.md)
- I want the app to be self-contained so scripts to run the VLM should also be included
- No gui required, happy to run from the CLI
- Caption text quality is paramount, but speed is also important
- Initially I want to try with a selection of models to evaluate them
- Can I evaluate with another LLM? Maybe a higher quality model on OpenRouter or Opus? Cost?
- Can the evaluation be automated?