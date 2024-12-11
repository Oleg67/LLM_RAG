#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  
#  Copyright 2024 OlehYudin <avis2329@gmail.com>
#  
#  
#  

from langchain_core.prompts import PromptTemplate

extract_topic = PromptTemplate.from_template(
            """
            You are an expert in analyzing scientific papers.
            You have received the text:
            ###
                {text}
            ------------- \n
            Your task is to carefully analyze this text. 
            And answer the question:
            What are the main subject areas of research and study in the text?
            Compare these subject areas with the names in the list:
            ###
               {topics}
            --------------\n
            Name more than one, but less than {N} names of these areas or topics. 
            If possible, use the names from the list, otherwise, 
            if the text contains a name of the subject of research that does not exist in the list, give a new name of the topic.
            
            The output format is a dictionary in valid JSON format.
            Where the keys are the names of the topics studied in the text, and the values are short descriptions of these topics.
            Example: 
                <Topic_title1 : short_description_1, 
                Topic_title2 : short_description_2, 
                ...
                    >
            """
                )
                                            
check_dublicates_topic = PromptTemplate.from_template(
            """
            You are given a dictionary with topics studied in scientific articles, key - topic name, value - short description of the topic:
            ###
                {topics}
            -----------
            Your task is to carefully analyze this dictionary, especially the topic descriptions, 
            and merge those topics that are paraphrases or near-duplicates of each other into a new general topic. 
            Give this topic a new name and a short description.

            The output format is a dictionary of renames in valid JSON format.
            Where the keys are STRICTLY the keys from the input dictionary, and the values ​​are the new topic names created by the merge, 
            with short descriptions.
            These new topic names may be duplicates, for example when two or more older topic names are merged into a single new name.
            
            Example: 
                <Old_Topic_name1 : <"name": New_Topic_name_1, "description": description of New_Topic_name_1>,
                Old_Topic_name2 : <"name": New_Topic_name_2, "description": description of New_Topic_name_2>,
                ...
                   >
            """
                )


check_topic = PromptTemplate.from_template(
            """
            You are given two lists of topics names , first list:
            ###
               {list1}
            -----------
            and second list:
            ###
               {list2}
            ----------
            
            Your task is to compare these lists and establish a one-to-one correspondence between them using the similarity of names.
            The names may not be exactly identical, but only similar.

            The output format is a dictionary in valid JSON format.
            Where the keys are the names from the first list, and the values are the names from the second list.
            
            Example: 
                <1stTopic_name : 2stTopic_name, ... >
            """
                )
                                   

clustering_topics_prompt = PromptTemplate.from_template(
            """
            You are an expert in analyzing scientific articles.
            Your task is to carefully analyze the topics given in this dictionary  studied in some articles.
            Key of the dictionary is a topic name, value is a short description of the topic:
            ###
                {topics} 
            -----------------------
            
            Group these topics into no more than {N} clusters, using their descriptions and names.
            Each cluster must contain topics with a similar name and their descriptions. 
            The number of topics in a cluster can be any.
            Give these clusters names and brief descriptions.
            Try not to create a new cluster if the same or similar cluster already exists.
            Each topic must be present in one and only one cluster.
            All topics from this dictionary must be present in clusters, i.e. cannot be topics that are not in any cluster.
            
            Sum the number of topics in all clusters, you should get {L}.
            If it is not true you should regroup the clusters.
            
            The output format is a dictionary in valid JSON format. The dictionary should contain:
                key — cluster name, value — dictionary with two keys,
                    key_1 = 'description' , value_1 = a cluster description, 
                    key_2 = 'nodes',  value_2 = list of nodes in the cluster with the same names 
                    and descriptions as in the input dictionary.
            
            Example:
                <"cluster_name_1": 
                        <"description": "description of cluster_1",
                        "nodes": [list of nodes in cluster_1 as dictionaries]>,
                "cluster_name_2": 
                        <"description": "description of cluster_2",
                        "nodes": [list of nodes in cluster_2 as dictionaries]>,     
                ...
                >
            """
                ) 
            
clustering_topics_add = PromptTemplate.from_template(
            """
            You are an expert in analyzing scientific articles.
            Your task is to carefully analyze two dictionary.
            First is a dictionary of clusters of topics: keys are names of clucters, values are short descriptions of clusters topics.
            That is a first dictionary:
            ###
               {cluster}
            ---------------------
            
            Second is a dictionary  of topics studied in some articles.
            Key of the dictionary is a topic name, value is a short description of the topic.
            That is a second dictionary:
            ###
               {topics} 
            -----------------------
            
            Add topics from second dictionary  into existed clusters from first dictionary if it is possible.
            Or group these topics from second dictionary into new clusters.
            
            Each new cluster must contain topics with a similar name and their descriptions. 
            The number of nodes in a cluster can be any.
            Give these clusters names and brief descriptions.
            Try not to create a new cluster if the same or similar cluster already exists.
            Each topic must be present in one and only one cluster.
            All topics from the second dictionary must be present in clusters, i.e. there cannot be topics that are not in any cluster.

            The output is a new dictionary of clusters.
            The first original dictionary is supplemented with new clusters, 
            or old clusters from the original dictionary are supplemented with new topics.
            The output format is a dictionary in valid JSON format. The dictionary should contain:
                key — cluster name, value — dictionary with two keys,
                    key_1 = 'description' , value_1 = a cluster description, 
                    key_2 = 'nodes',  value_2 = list of dictionaris < topic_name: short_description> as in the input dictionary.
            
            Example:
                <"cluster_name_1": 
                        <"description": "description of cluster_1",
                        "nodes": [list of nodes in cluster_1 as dictionaries]>,
                "cluster_name_2": 
                        <"description": "description of cluster_2",
                        "nodes": [list of nodes in cluster_2 as dictionaries]>,     
                ...
                >
            """
                )

extract_author = PromptTemplate.from_template(
            """
            You are an expert in analyzing scientific papers.
            You have received the text:
            ###
                {text}
            ------------- \n
            Your task is to analyze the given text and answer the question: 
            Who  are authors of the text and where was this study?
            If there is not  an author in the text, give me 'No author'. \n
            
            The output format is a dictionary in valid JSON format.
            Where keys is an author names and values is a his institution.
            Example: 
                <author_1: institute_1, 
                author_2: institute_2,
                ...
                >
            """
                )

is_topic = PromptTemplate.from_template(
            """
            You are an expert in analyzing scientific papers.
            Your task is to carefully analyze the given text:
            ###
               {text}
            --------------
            And a list subject areas of research or topics:
            ###
               {topics}
            --------------
            And give me an answer of the question:
            Which topics are studing in the text?
             
            The output format is a dictionary in valid JSON format.
            Where keys is a list of topics names and values True or False if the topic is studing in the text.
            Example: 
                <Topic_name1 : True, Topic_name2 : False, ... >
            """
                )                        



def main(args):
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
