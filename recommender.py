"""
Course Recommeder System
"""

from cgitb import enable
from cmath import cos
from soupsieve import escape
import streamlit as st
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt 
import altair as alt

from rake_nltk import Rake
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import word_similarity as ws
global jk

FILTERED_COURSES = None
SELECTED_COURSE = None

@st.cache(persist=True)
def clean_col_names(df, columns):
	"""
	Cleans column names
	-----
	columns:
		List of column names
	"""

	new = []
	for c in columns:
		new.append(c.lower().replace(' ','_'))
	return new

@st.cache(allow_output_mutation=True)
def load_data():
	source_path1 = os.path.join("udemydataset/weighted_clean_dataset.csv")
	df_overview = pd.read_csv(source_path1)
	return df_overview
	# df = pd.concat([df_overview, df_individual], axis=1)

	# # preprocess it now
	# df = prepare_data(df)

	# return df

@st.cache(persist=True)
def filter(dataframe, chosen_options, feature, id):
	
	# temp = filter(df, skills_select, 'Type', 'Link')
	selected_records = []
	for i in range(1000):
		for op in chosen_options:
			if op in dataframe[feature][i]:
				selected_records.append(dataframe[id][i])
	return selected_records

def extract_keywords(df, feature):
	r = Rake()
	keyword_lists = []
	for i in range(1000):
		descr = df[feature][i]
		r.extract_keywords_from_text(descr)
		key_words_dict_scores = r.get_word_degrees()
		keywords_string = " ".join(list(key_words_dict_scores.keys()))
		keyword_lists.append(keywords_string)
		
	return keyword_lists

def extract_keywords(df, feature):

	r = Rake()
	keyword_lists = []
	for i in range(df[feature].shape[0]):
		descr = df[feature][i]
		r.extract_keywords_from_text(descr)
		key_words_dict_scores = r.get_word_degrees()
		keywords_string = " ".join(list(key_words_dict_scores.keys()))
		keyword_lists.append(keywords_string)
		
	return keyword_lists

def recommendations(df, input_course, cosine_sim, find_similar=True, how_many=5):
	
	# initialise recommended courses list
	recommended = []
	selected_course = df[df['Title']==input_course]
	
	# index of the course fed as input
	idx = selected_course.index[0]
	# print(cosine_sim)
	# print(idx)
	# print(cosine_sim[idx])
	# cosine_sim = jk.hash_value
	# print(str(df.iloc[[idx]]['text_clean']))
	# zzzz = count.transform([df.iloc[[idx]]['text_clean']])
	# print(zzzz)
	cosine_sim_new = cosine_similarity(cosine_sim[idx],cosine_sim)

	# creating a Series with the similarity scores in descending order
	if(find_similar):
		score_series = pd.Series(cosine_sim_new[0]).sort_values(ascending = False)
	else:
		score_series = pd.Series(cosine_sim_new[0]).sort_values(ascending = True)
	score_series=score_series.to_frame()
	joined_df = score_series.join(df)
	
	# getting the indexes of the top 'how_many' courses
	if(len(joined_df) < how_many):
		how_many = len(joined_df)
	new_joined_df = joined_df.head(how_many)
	print("befor-sort",new_joined_df)
	if(find_similar):
		new_joined_df=new_joined_df.sort_values("weighted_rating",ascending = False)
	else:
		new_joined_df =new_joined_df.sort_values("weighted_rating",ascending = True)
	top_sugg = list(new_joined_df.iloc[1:how_many+1].index)
	print("aftersort,",new_joined_df)
	# populating the list with the titles of the best 10 matching movies
	for i in top_sugg:
		qualified = df['Title'].iloc[i]
		recommended.append(qualified)
		
	return recommended

def content_based_recommendations(df, input_course, courses):

	if input_course is None:
		st.write("Please select input course")
		return
	# filter out the courses
	df = df[df['Title'].isin(courses)].reset_index()
	# create Description keywords
	# df['descr_keywords'] = extract_keywords(df, 'Description')
	# instantiating and generating the count matrix
	count = TfidfVectorizer()
	count_matrix = count.fit_transform(df['text_clean'])
	# generating the cosine similarity matrix
	# cosine_sim = cosine_similarity(count_matrix, count_matrix)

	# make the recommendation
	rec_courses_similar = recommendations(df, input_course, count_matrix, True)
	temp_sim = df[df['Title'].isin(rec_courses_similar)]
	temp_sim =temp_sim.sort_values("weighted_rating",ascending = False)
	rec_courses_dissimilar = recommendations(df, input_course, count_matrix, False)
	temp_dissim = df[df['Title'].isin(rec_courses_dissimilar)]
	temp_dissim =temp_dissim.sort_values("weighted_rating",ascending = True)

	# top 3
	st.write("Top 5 most similar courses")
	# def make_clickable(link):
	# 	text = link
	# 	return f'<a target="_blank" href="{link}">{text}</a>'

	# temp_sim['Link'] = temp_sim['Link'].apply(make_clickable)
	# st.write(temp_sim.to_html(escape=False), unsafe_allow_html=True)
	st.write(temp_sim)
	st.write("Top 5 most dissimilar courses")
	st.write(temp_dissim)

def prep_for_cbr(df):

	# content-based filtering
	st.header("Content-based Recommendation")
	st.sidebar.header("Filter on Preferences")
	st.write("This section is entrusted with the responsibility of"
		" analysing a filtered subset of courses based on the **Types**"
		" of the courses. This filter can be adjusted on"
		" the sidebar.")
	st.write("This section also finds courses similar to a selected course"
		" based on Content-based recommendation. The learner can choose"
		" any course that has been filtered on the basis of their carrier type"
		" in the previous section.")
	st.write("Choose course from 'Select Course' dropdown on the sidebar")

	# filter by skills
	skills_avail = ['Finance', 'Tech', 'Design', 'Business', 'Marketing']
	# for i in range(1000):
	# 	skills_avail = skills_avail + df['skills'][i]
	skills_avail = list(set(skills_avail))
	skills_select = st.sidebar.multiselect("Select Your interests", skills_avail)
	# use button to make the update of filtering
	skill_filtered = None
	courses = None
	input_course = "Nothing"
	#if st.sidebar.button("Filter Courses"):
	temp = filter(df, skills_select, 'Type', 'Link')
	skill_filtered = df[df['Link'].isin(temp)].reset_index()
	# update filtered courses
	courses = skill_filtered['Title']
	st.write("### Filtered courses based on skill preferences")
	st.write(skill_filtered)
	# some more info
	st.write("**Number of programmes filtered:**",skill_filtered.shape[0])
	for j in skills_avail:
		st.write("**Number of " + j + " types courses**",
			skill_filtered[skill_filtered['Type']==j].shape[0])
	# basic plots
	chart = alt.Chart(skill_filtered).mark_bar().encode(
		y = 'course_provided_by:N',
		x = 'count(course_provided_by):Q'
	).properties(
		title = ''
	)
	# st.altair_chart(chart)

	# # there should be more than atleast 2 courses
	# if(len(courses)<=2):
	# 	st.write("*There should be atleast 3 courses. Do add more.*")

	input_course = st.sidebar.selectbox("Select Course", courses, key='courses')
	# use button to initiate content-based recommendations
	#else:
		#st.write("```Adjust the 'Select Skills' filter on the sidebar```")

	rec_radio = st.sidebar.radio("Recommend Similar Courses", ('no', 'yes'), index=0)
	if (rec_radio=='yes'):
		content_based_recommendations(df, input_course, courses)

	# recommend based on selected course

def main():

	st.title("Course Recommeder System")
	st.write("Exploring Courses on Udemy")
	st.sidebar.title("Set your Parameters")
	st.sidebar.header("Preliminary Inspection")
	st.header("About the Project")
	st.write("Course Recommeder System is a minimalistic system built to help learners"
		" navigate through the courses on Udemy, aided by a"
		" data-driven strategy. A learner could visualize different"
		" features provided in the dataset or interact with this app"
		" to find suitable courses to take. Course Recommeder System also can help"
		" identify suitable courses for a learner based on their"
		" learning preferences.")

	# load and disp data
	df = load_data()
	st.header("Dataset Used")
	st.write("For the purpose of building Course Recommeder System, data from Udemy"
		" dataset has been pulled out from kaggel opensource database which"
		" has around 44k cources available")
	st.markdown("Toggle the **Display raw data** checkbox on the sidebar"
		" to show or hide the processed dataset.")
	# toggle button to display raw data
	if st.sidebar.checkbox("Display raw data", key='disp_data'):
		st.write(df)
	else:
		pass
	st.markdown("### What does each feature represent?")
	st.write("**Title:** Name of the course")

	st.write("**Link:** URL to the course homepage")
	# Title					Type	Description
	st.write("**Type:** Is it a course, a professional certificate or a specialization?")
	st.write("**Summary:** Small Summery of the course")
	st.write("**Stars:** Overall rating of the course")
	st.write("**Rating:** Number of learners who rated the course")
	st.write("**Enrollment:** Number of learners enrolled")
	st.write("**Description:** About the course combination of all")
	# initiate CBR
	prep_for_cbr(df)
	
	
if __name__=="__main__":
	main()
