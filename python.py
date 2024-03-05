# Show Animated image
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


###################################### Home Page ######################################
if sel =='Home':
    st.write("##  Hi, Welcome to my project")
    st.title("Hindi Character Recognition")

    lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_4asnmi1e.json") # Link for animated image
    st_lottie(lottie_hello)

    st.info("The project's fundamental notion is that when it comes to constructing\
     OCR models for native languages, the accuracies achieved are rather low, and so this\
     is a sector that still need research. This model (as implemented here) can be extended\
     to recognize complete words, phrases, or even entire paragraphs.")

    st.info("Handwritten character recognition is an important area in the study of image processing\
     and pattern recognition. It is a broad field that encompasses all types of character recognition\
     by machine in a variety of application fields. The purpose of this pattern recognition area is to\
     convert human-readable characters into machine-readable characters. We now have automatic character\
     recognizers that assist people in a wide range of practical and commercial applications.")

###################################### Prediction Page ######################################
if sel =='Prediction':
    # lottie_pred = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_jz4fqbbk.json")
    # st_lottie(lottie_pred)

    file = st.file_uploader(" ")  # From here we can upload an image

    if file is None:
        st.write("### Please Upload an Image that contain Hindi Character")
    else:

        img = load_and_prep(file)

        # Display the uploaded image
        fig,ax = plt.subplots()
        ax.imshow(img.numpy().astype('uint8'))
        ax.axis(False)
        st.pyplot(fig)

        # Prediction for uploaded image
        pred_prob = model.predict(tf.expand_dims(img,axis=0))
        st.write("### Select the top n predictions")
        n=st.slider('n',min_value=1,max_value=5,value=3,step=1)

        class_name , confidense = get_n_predictions(pred_prob,n) # Top n prediction with class name and confidence(Probabilty)

        if st.button("Predict"):


            st.header(f"Top {n} Prediction for given image")

            # Horizontal bar chart for top n prediction with probability
            fig = go.Figure()

            fig.add_trace(go.Bar(
                    x=confidense[::-1],
                    y=class_name[::-1],
                    orientation='h'))
            fig.update_layout(height = 500 , width = 900, 
                        xaxis_title='Probability' , yaxis_title=f'Top {n} Class Name')

            st.plotly_chart(fig,use_container_width=True)


            st.success(f"The image is classified as \t  \'{class_name[0]}\' \t with {confidense[0]*100:.1f} % probability")

###################################### Download Page ######################################

if sel =='Get test images':
    st.write('# Download the test images')
    lottie_download = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_szdrhwiq.json") # Link for animated image
    st_lottie(lottie_download)

    # https://stackoverflow.com/questions/71589690/streamlit-image-download-button
    # How to make image download button, reference take from above link

    char_name_hindi = '# क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह ॠ त्र ज्ञ ० १ २ ३ ४ ५ ६ ७ ८ ९'.split() 
    for i in range(1,47):

        col1,col2 = st.columns(2)

        # Left side display the button for image download
        with col1:

            st.download_button(
                        label=f'Download the image of {char_name_hindi[i]}',
                        data = open(f'img/{i}.png', 'rb').read(),
                        file_name=f"{i}.png",
                        mime='image/png')

        # Right side display the original image
        with col2:

            img = Image.open(f'img/{i}.png')
            st.image(img)

###################################### Code Page ######################################

if sel =='Code':
    with open('app.py','r',encoding="utf8") as f:
        code = f.read()
    st.code(code,'python')
