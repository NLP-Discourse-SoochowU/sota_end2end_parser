# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""


class EDU:
    def __init__(self, edu, nlp=None):
        self.words, self.pos_tags = self.tok_analyse(nlp, edu)

    @staticmethod
    def tok_analyse(nlp, edu):
        tok_pairs = nlp.pos_tag(edu)
        words = [pair[0] for pair in tok_pairs]
        tags = [pair[1] for pair in tok_pairs]
        return words, tags
