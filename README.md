# YouTube Video Summarizer

The YouTube Video Summarizer is a Python script that allows you to extract video transcripts based on the video id, and summarize the 
text content using the GPT-3.5 Turbo model. You can apply the script to video of any language under the condition the video creator do 
not disable the transcript of the videos intentionally. By default video in German and English will be summarized in these languages, 
Summaries of videos spoken in all other languages will be in English. 


Summarizing long YouTube videos and interesting videos which have high number of viewers in different languages can offer several advantages:

- Time-saving: Summaries provide a concise overview of the video's content, allowing viewers to quickly grasp the main points without 
spending significant time watching the entire video. This is particularly beneficial for busy individuals who want to extract key 
information efficiently.

- Enhanced accessibility: Videos in different languages may have a language barrier for viewers who don't understand the spoken 
language. Summarizing these videos can help bridge the gap by providing a summarized version in a language that viewers understand, 
making the content more accessible and inclusive.

- Improved comprehension: Some videos, especially those covering complex or technical topics, may be challenging to understand fully. 
Summaries provide a simplified version of the content, breaking it down into key ideas and concepts. This aids comprehension, ensuring 
that viewers grasp the essential information.

- Multilingual support: YouTube is a global platform with a diverse audience. Summarizing videos in different languages caters to 
viewers from various language backgrounds, enabling them to access and understand the content better. This expands the reach and impact 
of the video, reaching a wider audience.

- Content curation: Summarizing popular videos with a high number of viewers helps curate and highlight the most relevant information. 
By distilling the main points, viewers can quickly evaluate whether the video aligns with their interests or needs. This can save time 
for viewers who are exploring multiple videos within a particular topic.


Overall, summarizing long YouTube videos and interesting videos with a high number of viewers in different languages offers time-saving 
benefits, accessibility improvements, enhanced comprehension, multilingual support, content curation, and knowledge sharing 
opportunities.

 ## Prerequisites

- `Python 3.x`
- `Google API Key` 
- `OpenAI API Key` 

## usage 
you have to pass your Youtube API Key and OpenAI API Key as parameters to the YouTubeVideoSummarizer constructor as following

summarizer = YouTubeVideoSummarizer(your Youtube API Key, your OpenAI API Key) 

then 

transcript,lang = summarizer.get_transcript(video_id)

and afterwards call 

summary = summarizer.summarize_text(transcript,lang)

and if you want to save it to a file ( text or audio file ) 

summarizer.write_to_textfile(video_id,summary)

or

summarizer.convert_to_audio(summary)


