# audio2text2summary

Create virtual environment

    python -m venv env

activate virtual environment
    source bin/activate

This streamlit app takes voice and returns text and the text summary using whisper and facebook/bart-large-cnn

Inside virtual environment clone the repository

    git clone https://github.com/Derngineer/audio2text2summary.git


Install requirements for the app

    pip install -r requirements.txt

run the following command to start the app

    streamlit run app2.py


Special attention
- To record your sound effectively ensure the soundwave is responsive.
- If not responsive stop the recording and start again
- Uploading files is pretty straight forward.
