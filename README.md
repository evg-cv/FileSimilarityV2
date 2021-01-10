Java 8. The command java -version should complete successfully with a line like: java version “1.8.0_92”.
wget http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip
cd stanford-corenlp-4.2.0
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
