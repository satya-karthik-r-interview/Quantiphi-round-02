# Quantiphi-round-02

Quantiphi interview round 02

## Instructions for colab notebook

just open the notebook `01_Q&A_notebook.ipynb` and run it as per the instructions provided inside.

## instructions for streamlit app

* Run with `GPU` enabled system
  
1. build the docker image using the build.sh script.

   ```bash
   bash docker/build.sh
   ```

2. run the docker compose file using the docker compose up command

    ```bash
    docker compose up -d
    ```

3. wait till it download the model then go to <http://localhost:8051> to use streamlit app.

you can ask anything from chapter 4 and 5.

### Note that if the model pull from hugggingface fails it may break the pipeline

use `docker compose logs fastapi` to confirm that the server is up.

### things taken care of

---
✅ Download the pdf from the link above

✅ To make indexing faster, you can pick any 2 chapters from the pdf and treat it as a source [chpter 4 and 5 picked based on the page numbers]

✅ Use any in-memory vector database if required.

✅ Use any open source HuggingFace model as the LLM Model

#### Output artifacts we need for evaluation

---

✅ Entire codebase in GitHub with links to access. (N/A)

✅ Please add docstrings wherever necessary.

✅ Additional Colab notebook to run the backend logic and evaluations: (this note book has everything that is required to run the logic)

✅ Please add text blocks in your Colab to add scenarios/assumptions etc to make it readable.

❌ Any additional artifacts like system design architecture, assumptions, list of issues you couldn't solve because of time constraints and how you can fix it in future.

#### Additional (bonus)

---
✅ Streamlit/Gradio Frontend to interact with your pipeline

✅ Wrap the entire application inside a docker container

✅ Draft and implement all the necessary APIs using FastAPI or any other python web framework of choice

✅ Produce alternative way to do the RAG without using any library like Langchain, LLamaIndex or Haystack
