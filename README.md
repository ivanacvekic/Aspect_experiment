# L2 learners’ online processing of grammatical aspect in temporarily ambiguous sentences in English

The current project deals with online processing of temporary ambiguous sentences during a self-paced reading (SPR) task. 
The experiment aims at looking if L2 learners use the L2 English aspect information incrementally and if the existence of 
grammaticalized aspect in their first language (L1) aids them in the process. The participants in the study are L1 German, 
L1 Croatian and L1 Spanish learners of L2 English that either instantiate aspect (Croatian, Spanish) or not (German).

In the aspect_participant.py script, the descriptive analysis of the participants is calculated and between-group differences 
are tested:
- Age
- Age of Acquisition (AoA)
- Years of learning English
- Test: English proficiency - LexTALE
- Test: English proficiency - Category Fluency Task (CFT)
- Test: offline knowledge of English aspect (Cloze test)
- Test: online knowledge of progressive and simple aspect - Self Paced Reading (SPR)

In the aspect_main_experiment.py script, the data is cleaned from outliers, it is visualized, and within-group and 
between-group differences are tested using the lmer model.
