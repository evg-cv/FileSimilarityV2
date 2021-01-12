# FileSimilarity_v2

## Overview
This project is to implement the several NLP functions using StanfordCoreNLP, Spacy, Gensim, NLTK, Scikit-learn, Pandas
libraries.
The core technology of this project is the semantic analysis of the text using pos-tagging, the search of availability 
of the keywords in the text, and the content beyond the master key.
Also, a pre-trained model which is part of a model that is trained on 100 billion words from the Google News Dataset is 
used for Natural Language Processing.

## Structure

- src

    The main source code for NLP tech.
    
- utils

    * The pre-trained model for NLP
    * The source code for management of the folders and files in this project and tool.
    
- app

    The main execution file

- requirements

    All the dependencies for this project
    
- settings

    Several settings including the input csv file path. 

## Installation

- Environment

    Ubuntu 18.04, Windows 10, Python 3.6, Java 1.8

- Java 8. The command java -version should complete successfully with a line like: java version “1.8.0_92”.

- Dependency Installation

    * Please go ahead to this project directory and run the following commands in the terminal
    ```
        pip3 install -r requirements.txt
        python3 -m spacy download en
        python3 -m nltk.downloader all
    ```
    * Please download StanfordCoreNLP by running the following command in the terminal.
    ```
       wget http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
       unzip stanford-corenlp-latest.zip 
    ``` 

- Please create the "model" folder in the "utils" folder of this project directory and copy the model into the "model" folder.
 
## Execution

- Please run the following command in the new terminal for StanfordCoreNLP server. 

    ```
        cd stanford-corenlp-4.2.0
        java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    ``` 
  
- Please set KEYWORD_FILE_PATH, SIMILARITY_FILE_PATH in settings file with the absolute path of the keyword csv file and 
similarity csv file which contains master keys and text iterations.
 
- Please run the following command in the terminal

    ```
        python3 app.py
    ```

- The output file will be saved in the output folder.

## Note
- In Windows, when you set the path, you have to replace "\" with "\\".

- When creating the input csv files, please refer the sample files (keyword_sample.csv & similarity_sample.csv) and follow
their field names without any modification.
