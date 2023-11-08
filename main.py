import streamlit as st
import os 
import openai
from htmlTemplates import css, bot_template
from langchain.chains import LLMChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from streamlit_option_menu import option_menu

model_engine = "text-davinci-003"

# Define a function to handle the translation process
def translate_text(text, target_language):
    # Define the prompt for the ChatGPT model
    prompt = f"Translate '{text}' to {target_language}"
    
    # Generate the translated text using ChatGPT
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    # Extract the translated text from the response
    translated_text = response.choices[0].text.strip()
    
    return translated_text

st.set_page_config(page_title="Voice GPT",layout="wide")
with st.sidebar:
     selected = option_menu("Voice GPT", ["About","Transcribe", 'Translate'], 
        icons=['book', 'activity', "arrow-bar-right"], menu_icon="cast", default_index=1,styles={
        "container": {"padding": "0!important", "background-color": "#fafafa", "font-family": "Permanent Marker"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#ppp"},
        "nav-link-selected": {"background-color": "lightblue"},
    })

if selected == "About":

    st.title(":Orange[About]")
 
    st.markdown(''':rainbow[Welcome to Voice GPT] ''')


    st.markdown(""" 
                "Voice GPT" typically refers to a technology that combines the power of OpenAI's GPT models with speech recognition and synthesis capabilities. This allows for natural language understanding and generation from spoken language.   
                :rainbow[Key Features:]  
                :rainbow[Transcription Services:]     
                Voice GPT can transcribe spoken content into written text, making it a valuable tool for businesses, journalists, and researchers who need accurate and efficient transcriptions of audio content.  
                :rainbow[Translation Services:]    
                Voice GPT can assist in language translation, making it easier for people to communicate across language barriers.  
                :rainbow[Try your Self:]     
                It's as simple as uploading your audio file or Record your audio and generate your transcription within minutes.   
                This tool also provides the summary of the Transcription and the Sentiment Label for the Transcription.  
                You can try Tranlate Service as well where you can provide Text in English and select the language to be Translated  
                and have your results.""",unsafe_allow_html=True)
     
if selected == "Transcribe":

    def main():
    #st.set_page_config(page_title="Voice GPT")
        st.write(css, unsafe_allow_html=True)
    st.session_state.setdefault("audio_file_path", None)
    st.session_state.setdefault("transcript", None)
    st.session_state.setdefault("transcript_summary")
    st.session_state.setdefault("sentiment_label")
    st.session_state.setdefault("sentiment_results")
    st.title(":Orange[Transcribe Service]")
    st.write('Select Below Options to Perform Transcription')


    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    upload_mode = st.radio("Upload Mode", options=['Audio Recording Upload', 'Voice Record'])

    if upload_mode == 'Audio Recording Upload':
         uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'mp4', 'mpeg', 'mpga', 
                                                                'm4a', 'wav', 'webm'])
         if uploaded_file is not None:
                    audio_bytes = uploaded_file.read()
                    st.audio(audio_bytes, format="audio/wav")
    if upload_mode == 'Voice Record':
                audio_bytes = audio_recorder()
                if audio_bytes:
                    file_path = os.path.join(upload_dir, 'audio_record.wav')
                    with open(file_path, 'wb') as fp:
                        fp.write(audio_bytes)
                    st.audio(audio_bytes, format="audio/wav")
                    uploaded_file = file_path

    if st.button("Generate Transcript/Summary") and uploaded_file:
        with st.spinner('Transcribing...'):
            if isinstance(uploaded_file, str):
                        st.session_state.audio_file_path = uploaded_file
            else:
                        file_path = os.path.join(upload_dir, uploaded_file.name)
                        with open(file_path, 'wb') as f:
                             f.write(uploaded_file.getbuffer())
                        st.session_state.audio_file_path = file_path
                        
            with open(st.session_state.audio_file_path, 'rb') as audio_file:
                st.session_state.transcript = openai.Audio.transcribe("whisper-1", audio_file)['text']
            summary_prompt = PromptTemplate(
                    input_variables=['input'],
                    template='''
                    Summarize this audio transcript: 
                    <transcript>{input}</transcript>
                    '''
                )
            sentiment_prompt = PromptTemplate(
                        input_variables=['transcript','summary'],
                        template='''
                            Return a single word sentiment of either ['positive','negative' or 'neutral'] from this transcript and summary.
                            \nTRANSCRIPT: {transcript}
                            \nTRANSCRIPT SUMMARY: {summary}
                            \nSENTIMENT LABEL HERE ('positive','negative', or 'neutral') <comma-seperated> REPORT HERE:
                        '''
                    )
            llm = OpenAI(temperature=0.65, model_name="gpt-4")
            summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
            st.session_state.transcript_summary = summary_chain.run(input=st.session_state.transcript)
            sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
            st.session_state.sentiment_results = sentiment_chain.run(transcript=st.session_state.transcript,
                                                                           summary=st.session_state.transcript_summary).split(",")
            st.session_state.sentiment_label = st.session_state.sentiment_results[0]
            st.session_state.sentiment_report = "".join(st.session_state.sentiment_results[1:])

    if st.session_state.audio_file_path:
                if st.session_state.transcript:
                    st.subheader(st.session_state.audio_file_path.split("\\")[1])
                    with st.expander("Transcription", expanded=True):
                        st.write(st.session_state.transcript)
                    if st.session_state.transcript_summary:
                        with st.expander("Summary", expanded=True):
                            st.write(st.session_state.transcript_summary)
                        with st.expander("Sentiment Analysis", expanded=True):
                            st.write(f"Sentiment Label: {st.session_state.sentiment_label}")
                            #st.write(f"Sentiment Report: {st.session_state.sentiment_report}")
                        #with st.expander("Text Statistics", expanded=True):
                            #st.write(f"Transcription Word Count: {len(st.session_state.transcript.split())}")
                            #st.write(f"Transcription Character Count: {len(st.session_state.transcript)}")
    if __name__ == "__main__":
     load_dotenv()
    #openai.api_key = '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'
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
if selected == "Translate":

    def main():
    # Set up the Streamlit UI
        st.title(":Orange[Translate Service]")
        #st.sidebar.header('Language Translation App')
        st.write('Enter text to translate and select the target language:')
    
    # Create a text input for the user to enter the text to be translated
        text_input = st.text_input('Enter text')
    
    # Create a selectbox for the user to select the target language
        target_language = st.selectbox('Select language', ['Arabic', 'English', 'Spanish', 'French', 'German', 'Telugu', 'Hindi'])
    
    # Create a button that the user can click to initiate the translation process
        translate_button = st.button('Translate')
    
    # Create a placeholder where the translated text will be displayed
        translated_text = st.empty()
    
    # Handle the translation process when the user clicks the translate button
        if translate_button:
          translated_text.text('Translating...')
          translated_text.text(translate_text(text_input, target_language))
          st.write("Translation Work is in Progress")

    if __name__ == '__main__':
     main()


    
