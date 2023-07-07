import streamlit as st
import pandas as pd
import scipy.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sb
import plotly.offline as py
from Plotvib import make_segments, convert_df, plot_fft
from Plotvib import plot_data, filter_freq, plot_box
from Plotvib import extract_features, plot_features, envelope_plot, extract_features_time
import scipy.fft as sp
import plotly.colors
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit



plotly_colors = [
    'rgb(31, 119, 180)',  # Blue
    'rgb(255, 127, 14)',  # Orange
    'rgb(44, 160, 44)',   # Green
    'rgb(214, 39, 40)',   # Red
    'rgb(148, 103, 189)', # Purple
    'rgb(140, 86, 75)',   # Brown
    'rgb(227, 119, 194)', # Pink
    'rgb(127, 127, 127)', # Gray
    'rgb(188, 189, 34)',  # Olive
    'rgb(23, 190, 207)',  # Teal
    'rgb(140, 140, 140)', # Dark Gray
    'rgb(255, 187, 120)', # Light Orange
    'rgb(44, 50, 180)',   # Navy Blue
    'rgb(214, 39, 40)'    # Red (same as above for emphasis)
]

# Initialize the app
# Configure page settings
extracted_files = []
extracted_file_names = []

st.set_page_config(layout="wide")
box = st.container()


# helper functions
def check_file(files, str):
            for file in files:
                if not file.endswith(str):
                    return False
            return True


def read_mat(file):
    data = scipy.io.loadmat(file)
    return data


def read_csv(file):
    data = pd.read_csv(file)
    return data


def process_file(file, col):
    file_extension = file.name.split(".")[-1]

    if file_extension == "mat":
        data = read_mat(file)
        keys = list(data.keys())
        filtered_keys = [key for key in keys if "_DE_time" in key]
        default_key = filtered_keys[0] if filtered_keys else None

        if len(keys) > 0:

            selected_key = col.selectbox("Select a key", keys,index=keys.index(default_key) if default_key else 0)
            selected_data = np.array(data[selected_key]).reshape(-1)
            return { f"{file.name.split('.')[0]}": selected_data}
        
        else:
            col.write("MAT File does not contain any keys.")

    elif file_extension == "csv":
        data = read_csv(file)
        if int(data.shape[1]) == 2:
            selected_data_x = np.array(data.iloc[:, 0]).reshape(-1)
            selected_data_y = np.array(data.iloc[:,1]).reshape(-1)
            return {f"{ file.name.split('.')[0]}_x": selected_data_x,
                    f"{file.name.split('.')[0]}_y" : selected_data_y,
                    }
        else:
             selected_data = np.array(data.iloc[:, 0]).reshape(-1)
             return {f"{ file.name.split('.')[0]}": selected_data}

    else:
        col.error(
            f"Invalid file format: {file_extension}. Only MAT files are supported."
        )


def evaluate_fitness(features, Y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features,Y, test_size=0.2, random_state=42)
    
    # Train a classifier using the selected features
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    
    # Evaluate the classifier's accuracy on the test set
    accuracy = classifier.score(X_test, y_test)
    
    return accuracy

def geneticAlgo(features,Y):
    
    
    N = features.shape[1]
    chromosome_length = N 
    
    # Step 3: Initial Population
    population_size = 50
    population = np.zeros((population_size, chromosome_length), dtype=int)
    for i in range(population_size):
        ones_indices = np.random.choice(chromosome_length, 5, replace=False)
        population[i, ones_indices] = 1
        
    # Step 4: Fitness Evaluation
    fitness_scores = []
    for chromosome in population:
        # Convert binary chromosome to selected features
        
        selected_features = features.iloc[:, np.nonzero(chromosome)[0]]
        fitness = evaluate_fitness(selected_features,Y)
        fitness_scores.append(fitness)
        
    max_generations = 1
    generation = 0
    
    while generation < 2:
        print("Entered")
        # Step 5: Selection
        # Perform tournament selection to choose parents
        num_parents = int(population_size / 2)
        parents = []
        for _ in range(num_parents):
            tournament_size = 5
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_index])

        offspring = []
        for i in range(0, num_parents, 2):
            if i != 24:
                parent1 = parents[i]
                parent2 = parents[i+1]
                crossover_point = np.random.randint(1, chromosome_length - 1)
                child1 =np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                offspring.extend([child1, child2])
            else:
                
                offspring.append(parents[i])
                
        mutation_rate = 0.01
        for i in range(num_parents):
            if np.random.rand() < mutation_rate:
                random_index = np.random.randint(chromosome_length)
                offspring[i][random_index] = 1 - offspring[i][random_index]
        
        population[:num_parents] = parents
        population[num_parents:] = offspring
        
        fitness_scores = []
        for chromosome in population:
            selected_features = features.iloc[:,np.nonzero(chromosome)[0]]
            fitness = evaluate_fitness(selected_features,Y)
            fitness_scores.append(fitness)
        generation += 1
        print(generation)
        
    best_chromosome = population[np.argmax(fitness_scores)]
    selected_features = features.iloc[:,np.nonzero(best_chromosome)[0]]
    
    return selected_features

def plot_time(
    extracted_files, 
    index,
    div, 
    total_segment, 
    seg_num ,
    env_plot, isolated,
    sampling_frequency
):
    figs = []
    fig = None
    selected_name =   list(index)
    for k in selected_name:
            a = make_segments(extracted_files[k], total_segments=total_segment)
            if env_plot:
                fig = envelope_plot(df=a, title=k, seg_num=seg_num, show_real= (not isolated), sampling_freq = sampling_frequency)
                figs.append(fig) 
            else:
                fig = plot_data(df=a, title=k, seg_num=seg_num , sampling_freq = sampling_frequency)
                figs.append(fig)

    default_colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    combined_fig = go.Figure()
    if not env_plot:
        for i, fig in enumerate(figs):
            for  trace in fig.data:
                combined_fig.add_trace(trace.update(line=dict(color=default_colors[i]),showlegend=True, name =trace.name)  )
    else:
        for i, fig in enumerate(figs):
                if not isolated:
                    combined_fig.add_trace(fig.data[0].update(line=dict(color=default_colors[i]) , showlegend= False, name =fig.data[0].name))
                combined_fig.add_trace(fig.data[-1].update(line=dict(color= plotly_colors[-(i + 1)]), showlegend= True, name =fig.data[-1].name))

   
    combined_fig.update_layout(
        title="Time Domain",
        title_x=0.4,
        xaxis_title="Time",
        yaxis_title="Amplitude",
        legend_title="Legend",
        width= 1000,
        plot_bgcolor='white',
        showlegend=True,
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        margin=dict(l=50, r=20, t=80, b=0),
       )
    
    div.plotly_chart(combined_fig)
    div.markdown("---")

        
def plot_freq(
    extracted_files,
    index,
    filtered_freq,
    div,
    freq_range,
    total_segments,
    seg_num,
    limit,
    sampling_frequency,
):
    
    selected_name = [index]
    frequency = abs(sp.fftfreq(( extracted_files[selected_name[-1]].shape[0] // total_segments), 1 / sampling_frequency))
    steps = frequency[1] - frequency[0]
   
  
    for k in selected_name:
                
                a = make_segments(extracted_files[k], total_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, total_segments=total_segments)
                if filtered_freq:
                    c = filter_freq(b, amp=limit)
                    fig = plot_fft(
                            x_axis=frequency[: int(freq_range * (steps**-1))],
                            y_axis=c,
                            title=k,
                            seg_num=seg_num,
                            )
                    div.plotly_chart(fig)

                else:
                    fig = plot_fft(
                        x_axis=frequency[: int(freq_range * (steps**-1))],
                        y_axis=b,
                        title=k,
                        seg_num=seg_num,)
                    
                    div.plotly_chart(fig)

  
def plot_feat(
    extracted_files,
    index,
    filtered_freq,
    stats_features,
    div,
    total_segments,
    sampling_freq,
):
    keys = stats_features
    selected_name = list(index)
    # name = [extracted_file_names[0][k] for k in ix]
    extracted = {}
    try:
        for k in selected_name:
                a = make_segments(extracted_files[k], total_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, total_segments=total_segments)
                if filtered_freq:
                     c = filter_freq(b,file_name=k)
                     d = extract_features(c)
                     extracted[k] = d
                else:
                    d = extract_features(b)
                    extracted[k] = d   
    except:
        st.error("Please select a right key")
    
    if len(keys) > 0:
        div.plotly_chart(plot_features(extracted, keys=[f"{keys}"]))
    else:
        div.plotly_chart(plot_features(extracted))


def box_plot(
    extracted_files,
    index,
    filtered_freq,
    stats_features,
    div,
    total_segments,
    sampling_freq,
):
    keys = stats_features.lower()
    selected_name = list(index)
    extracted = {}
    try:
        for k in selected_name:
                a = make_segments(extracted_files[k], total_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, total_segments=total_segments)
                if filtered_freq:
                     c = filter_freq(b,file_name=k)
                     d = extract_features(c)
                     extracted[k] = d
                else:
                    d = extract_features(b)
                    extracted[k] = d   
    except:
        st.error("Please select a right key")

    if keys:
        div.plotly_chart(plot_box(extracted, value=keys))
    else:
        div.plotly_chart(plot_box(extracted, value="mean"))


def plot_scatter(
    extracted_files,
    index,
    div,
    total_segment,
    sampling_frequency,
    seg_num=1,
):
    combined_fig = go.Figure()
    selected_name = list(index)
    default_colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    for i, k in enumerate(selected_name):
            a = make_segments(extracted_files[k], total_segments=total_segment)
            combined_fig.add_trace(go.Scatter(
                                 y=a[seg_num - 1],
                                 mode = 'markers',
                                 marker=dict(color=default_colors[i]),
                                 name=f'{k} |  Segment {seg_num}',
                                 showlegend=True))
                                
        
   
    combined_fig.update_layout(
        title="Time Domain",
        title_x=0.4,
        xaxis_title="Data Points",
        yaxis_title="Amplitude",
        legend_title="Legend",
        width= 1000,
        plot_bgcolor='white',
        showlegend=True,
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        margin=dict(l=50, r=20, t=80, b=0),
       )
    
    div.plotly_chart(combined_fig)
    div.markdown("---")


def plot_lcurve(estimator, X, y):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }

    display = LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax)
    display.plot(ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.set_title(f"Learning Curve for {estimator.__class__.__name__}")
    fig.set_size_inches(8, 6) 
    return fig

from sklearn.metrics import roc_curve, auc

def plot_roc(model, X_test, y_test,fs_method,le):
    y_pred_prob = model.predict_proba(X_test)
    n_classes = len(model.classes_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=model.classes_[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Create the figure
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green', 'orange', 'purple']  # You can add more colors for more classes
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve for class {} (area = {:.2f})'.format(
            list(model.classes_)[i] if fs_method != "ReliefF" else list(le.classes_)[i], roc_auc[i]))
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

    return fig


def plot_cm(model, X_test, y_test, fs_method, le):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=list(model.classes_) if fs_method != 'ReliefF' else list(le.classes_))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.grid(False)
    cm_display.plot(ax=ax)

    return fig

def model_complie(df,model,fs_method,div,analyze, super_label):
     
     from sklearn.tree import DecisionTreeClassifier
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.neighbors import KNeighborsClassifier
     from sklearn.svm import SVC
     from sklearn.linear_model import LogisticRegression
     from sklearn.metrics import accuracy_score, classification_report
     from sklearn.preprocessing import StandardScaler, LabelEncoder
     
     models = {
      "Logistic Regression" :  LogisticRegression(random_state=0, max_iter=100000),
      "SVM": SVC(probability=True),
      "KNN" :  KNeighborsClassifier(n_neighbors=5),
      "Decision Tree" : DecisionTreeClassifier(),
      "Random Forest" :  RandomForestClassifier(n_estimators=50)
     }

     num_segments = 10
     sampling_frequency = 12000
     frame_size = int(df[list(df.keys())[0]].shape[0] // num_segments)
     hop_length = int( frame_size//2)
     #st.write(df[list(df.keys())[0]].shape[0])
    # st.write(frame_size)
    # st.write(hop_length)

     df_features = {}
     for file in list(df.keys()):
         df_features[file] = extract_features_time(df[file], frame_size, hop_length)

     DF = {}
     length = None
     for keys  in df_features[list(df.keys())[0]].keys():
              DF[keys] = []
     DF['labels'] = []
     DF['super_labels'] = []
     length = len(df_features[file]['mean'])

     #st.write(length)
     #st.write(df_features)
     for file in list(df.keys()):
         DF['labels'].extend( [file[0]] * length  )
         file_ = file.split(' ')[0]
         DF['super_labels'].extend( [f'{file_[0]}{file_[-4:]}'] * length )
         for key in df_features[list(df.keys())[0]].keys():
              DF[key].extend(df_features[file][key])

     #st.write(DF)

     st.markdown(
                                  """
                                       <style>
                       .container-with-margin {
                        margin-top: -10px;
                        margin-bottom: 10px;
                                     }
                                </style>
                               """,
    unsafe_allow_html=True
)
     dataframe = pd.DataFrame(DF)
     X = dataframe.drop(columns=['labels', 'super_labels'])
     if super_label:
        Y =     dataframe.iloc[:,-1]
     else:
        Y =  dataframe.iloc[:,-2]

     st.write(dataframe)
     X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.25, random_state=42)
     from sklearn.feature_selection import SelectKBest, f_classif
     le = LabelEncoder()
     if fs_method == 'F-Score':
         selector = SelectKBest(f_classif, k=5)
         X_train = selector.fit_transform(X_train, y_train)
         X_test = selector.transform(X_test)

     if fs_method == 'OMP':
         from sklearn.preprocessing import StandardScaler, LabelEncoder
         le = LabelEncoder()
         scaler = StandardScaler()
         X_std = scaler.fit_transform(X)
         y = le.fit_transform(Y)
         from sklearn.linear_model import OrthogonalMatchingPursuit
         omp =  OrthogonalMatchingPursuit(n_nonzero_coefs=5)
         omp.fit(X_std,y)
         params = abs(omp.coef_)
         X_train = X_train.drop(columns=list(X.columns[params==0]))
         X_test =  X_test.drop(columns=list(X.columns[params==0]))

     if fs_method == 'MRMR':
         from sklearn.preprocessing import StandardScaler, LabelEncoder
         le = LabelEncoder()
         y = le.fit_transform(Y)
         from mrmr import mrmr_classif
         selectd_features = mrmr_classif(X,y, K = 5)
         X_train = X_train.loc[:,selectd_features]
         X_test =  X_test.loc[:,selectd_features]

     if fs_method == 'PCA':
         from sklearn.preprocessing import StandardScaler, LabelEncoder
         le = LabelEncoder()
         y = le.fit_transform(Y)
         from sklearn.decomposition import PCA
         pca = PCA(n_components=5)
         X_train = pca.fit_transform(X_train, y_train)
         X_test =  pca.transform(X_test)

     if fs_method == 'ReliefF':
         from sklearn.preprocessing import StandardScaler, LabelEncoder
         scaler = StandardScaler()
         X_std = scaler.fit_transform(X)
         le = LabelEncoder()
         y = le.fit_transform(Y)
         X_train, X_test, y_train, y_test = train_test_split(X_std,y, test_size = 0.25, random_state = 0)
         import sklearn_relief as sr
         r = sr.RReliefF(n_features = 5)
         X_train = r.fit_transform(X_train ,y_train)
         X_test = r.transform(X_test)

     if fs_method == 'Genetic Algorithm':
        X_new = geneticAlgo(X,Y)
        X_train, X_test, y_train, y_test = train_test_split( X_new, Y, test_size=0.25, random_state=42)
    

     model = models[model]
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
    
     if analyze == "Classification Report":
        report = classification_report(y_test, y_pred)
        modified_report = report.replace('precision', 'class \t \tprecision').replace('recall', 'recall').replace('f1-score', 'f1-score')

        with div:
           
            st.markdown('<div class="container-with-margin">', unsafe_allow_html=True)

            st.code(modified_report, language='plaintext')
            st.markdown('</div>', unsafe_allow_html=True)
    

     if analyze == "Confusion Matrix":
        fig = plot_cm(model,X_test, y_test,fs_method,le)
        with div: 
  
                buffer = BytesIO()
                fig.savefig(buffer, format='png')
                buffer.seek(0)
                st.markdown('<div class="container-with-margin">', unsafe_allow_html=True)
                st.image(buffer, width=600)
                st.markdown('</div>', unsafe_allow_html=True)
    
 
     if analyze == "Learning Curve":
         fig = plot_lcurve(model, X, Y)
         with div: 

            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.markdown('<div class="container-with-margin">', unsafe_allow_html=True)
            st.image(buffer, width=600)
            st.markdown('</div>', unsafe_allow_html=True)
        
     if analyze == "ROC Curve":
        fig = plot_roc(model, X_test, y_test,fs_method, le)
        with div: 
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.image(buffer, width=600)


#############################-----Enigne Starts Here-----#####################################


def main():

    df_seg ={}

    col_l, col_2 = box.columns([0.4, 0.6], gap="large")
    col_m1, col_m2 = box.columns([0.4, 0.6], gap="large")
    col_l.title("File Upload")
    col_r = col_2.container()
    col_r2 = col_2.container()
   
    

    
    uploaded_files = col_l.file_uploader(
    
        "Upload MAT or CSV files", accept_multiple_files=True
    )
    
    exp = col_l.expander("Select Keys", expanded=True)
    cols = [] # columns for each file key
    li = [] # list of file names
    r = [] # list of keys

    if uploaded_files:
        all_files = [file.name for file in uploaded_files]

        # check if all files are .mat or .csv
        if check_file(all_files, ".mat"):
            if len(uploaded_files) <= 4 and len(uploaded_files) > 0:
                cols = exp.columns(len(uploaded_files))

            else:
                i = len(uploaded_files) // 4 + 1
                for _ in range(i):
                    cols.extend(exp.columns(4))

            for file, col1 in zip(uploaded_files, cols):
                dic = process_file(file, col1)
                df_seg.update(dic)

        elif check_file(all_files, ".csv"):
            for file in  uploaded_files :
                dic = process_file(file, col_l)
                df_seg.update(dic)
        
        else:
            st.error("Please upload only CSV or Mat files.")

    # Getting all file names
    extracted_file_names.append(li)
    extracted_file_names.append(r)
    exp2 = col_l.expander("About Data", expanded=True)
    shape = df_seg[list(df_seg.keys())[0]].shape[0] if uploaded_files else None
    if uploaded_files:
         selected_shape = exp2.number_input("Total Data points", 1, int(shape), int(shape))

         for key in df_seg.keys():
            df_seg[key] = df_seg[key][:selected_shape]
    
    # MainWWork here
    if uploaded_files:

         # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        # Total Number of Segments : Slider
        total_segments      =  exp2.number_input("Total Number of Segments", 1, int(10e6), 10)

        # Sampling Frequency : Slider
        sampling_frequency  =  exp2.number_input("Sampling Frequency",       1, int(10e6), 12000)

        col3, col4  =  col_l.columns(2)

        domain = col3.selectbox( "What to plot", ["", "Time Domain", "Frequency Domain", "Features", "Box Plot", "Scatter Plot"], )
         # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


        #------------   Time  ----------------------#
        if domain == "Time Domain":

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

            # File Name : MultiSelect Box
            selected_file_name = col4.multiselect( "Select a file", list(df_seg.keys()) )

            # Envelope Plot : Check Box
            env_plot = col4.checkbox("Plot Envelope")
           
            # Segment Number : Slider
            seg_num = col3.number_input("Segment Number", 1, total_segments)

            # Isolate Envelope : Check Box to isolate envelope
            isolated = col4.checkbox("Isolate Envelope") if env_plot else None

           # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


            if selected_file_name:

                # Call plot_time function
                plot_time(
                    df_seg,
                    selected_file_name,
                    col_r,
                    total_segments,
                    seg_num,
                    env_plot,
                    isolated,
                    sampling_frequency
                )
            else:

                st.write("No file selected")

        #------------  Frequency   -----------------#
        if domain == "Frequency Domain":

             # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
            # File Name : Select Box
            selected_file_name = col4.selectbox( "Select a file", list(df_seg.keys()))

            # Frequency Range : Slider  
            frequency = abs(sp.fftfreq(( df_seg[selected_file_name].shape[0]// total_segments), 1 / sampling_frequency))

            # Steps : Slider
            steps = frequency[1] - frequency[0]

            # Segment Number : Number Input
            seg_num = col4.number_input("Segment Number", 1, total_segments)

            # Filter Frequency : Check Box
            filtered_freq = col4.checkbox("Filter Frequency")

            # Frequency Limit : Number Input
            limit = col4.number_input("Frequency Limit", 0.0,1.0,0.2,step = 0.01) if filtered_freq else None

            # Frequency Range : 
            freq_range = col3.slider( "Frequency Range", float(0), float(sampling_frequency/2),float(sampling_frequency/2),  step= steps)
             # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

            if selected_file_name:


                plot_freq(
                    df_seg,
                    selected_file_name,
                    filtered_freq,
                    col_r,
                    freq_range,
                    total_segments,
                    seg_num,
                    limit,
                    sampling_frequency,
                )

            else:
                col_l.write("No file selected")

        #------------  Features  -----------------#
        if domain == "Features":

            selected_file_name = col4.multiselect( "Select file(s)", list(df_seg.keys()))
            # multiselct stats features but only 4 can be selected at a time

            stats_features = col4.selectbox( "Select stats features",
                [
                    "Mean",
                    "Max",
                    "Variance",
                    "Skewness",
                    "Kurtosis",
                    "shape_factor",
                    "impulse_factor",
                ],
                key="stats_features",
            )

            filtered_freq = col4.checkbox("Filter Frequency")
            # multiselect time features but only 4 can be selected at a time

            if selected_file_name:
                
                plot_feat(
                    df_seg,
                    selected_file_name,
                    filtered_freq,
                    stats_features,
                    col_r,
                    total_segments,
                    sampling_frequency,
                )
            else:
                st.write("No file selected")

        #------------  Box Plot  -----------------#
        if domain == "Box Plot":

            selected_file_name = col4.multiselect( "Select file(s)", list(df_seg.keys()))

            # multiselct stats features but only 4 can be selected at a time
            stats_features = col4.selectbox( "Select stats features",  [ "Mean", "Max", "Variance", "Skewness", "Kurtosis", "shape_factor","impulse_factor"],)
            
            filtered_freq = col4.checkbox("Filter Frequency")
            # multiselect time features but only 4 can be selected at a time

            if selected_file_name:
                

                box_plot(
                    df_seg,
                    selected_file_name,
                    filtered_freq,
                    stats_features,
                    col_r,
                    total_segments,
                    sampling_frequency,
                )
            else:
                st.write("No file selected")

        #------------ Scatter Plot -----------------#
        if domain == "Scatter Plot":

            # multiselct stats features but only 4 can be selected at a time
            selected_file_name = col4.multiselect( "Select file(s)", list(df_seg.keys()))

            if selected_file_name:

               
                
                plot_scatter(
                    extracted_files=df_seg,
                    index=selected_file_name,
                    div = col_r,
                    total_segment=total_segments,
                    sampling_frequency=sampling_frequency,
                    seg_num=1,
                )
            else:
                st.write("No file selected")

        with col_l:
            if col_l.button("Show Files"):
                st.write("Extracted Files:", df_seg)
                st.write("Extracted File Names:", {key : val.shape for key, val in df_seg.items()})


        cont1, cont2 = col_m1.container().columns([0.5,0.5])
        
        model = None
        with col_m1:
             col_m1.markdown('Model')             
             model = cont1.selectbox( "Select a Model", ["", "Random Forest", "Decision Tree", "KNN", "SVM", "Logistic Regression"],)
        
        if model:
            feature_selection = cont1.checkbox("Feature Selection")
            super_label = cont1.checkbox("Super_label")
            fs_method = cont2.selectbox("Feature Selection Method",["",'F-Score','ReliefF','OMP',"PCA","Genetic Algorithm"] ) if feature_selection else None
            report = cont2.selectbox("Analyze",["Classification Report", "Confusion Matrix", "Learning Curve", "ROC Curve"])

            model_complie(
                 df=df_seg,
                 model=model,
                 div = col_m2 ,   
                 fs_method=fs_method,
                 analyze= report,
                 super_label = super_label,
                )     
        


if __name__ == "__main__":
    main()
