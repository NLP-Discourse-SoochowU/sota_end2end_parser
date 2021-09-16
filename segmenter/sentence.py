# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from stanfordcorenlp import StanfordCoreNLP
from segmenter.edu import EDU
path_to_jar = 'stanford-corenlp-full-2018-02-27'
nlp = StanfordCoreNLP(path_to_jar)


class Sentence:
    def __init__(self, sentence, edus_list):
        self.edus = self.build_edus(edus_list)
        self.sentence_txt = sentence
        self.dependency = self.gen_dependency()

    @staticmethod
    def build_edus(edus_list):
        edus_ = list()
        for edu in edus_list:
            edus_.append(EDU(edu, nlp))
        return edus_

    def gen_dependency(self):
        dep = nlp.dependency_parse(self.sentence_txt)
        return dep
