import numpy as np
import pandas as pd
import re
import gensim
import gensim.corpora as corpora
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show, output_file
from sklearn.metrics import pairwise_distances
from sklearn import manifold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from GPyM_TM import GSDMM

cols = ["#d62728", "#17becf", '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#1f77b4']

stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}


def clean_text(text):
    # Make lower case
    text = str(text).lower()
    # Remove stop words
    text = re.sub(r'\b(' + r'|'.join(stop_words) + r')\b\s*', '', text)
    # Remove .*?
    text = re.sub('[.*?]', '', text)
    # remove special chars
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub('\n', '', text)
    return text


def sent_to_words(sentences):
    for sentence in sentences:
        # Remove punctuation
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def latent_dirichlet_allocation(corpus, id2word, data_words):
    # Build the lda model
    latent_dirichlet_allocation_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                                        id2word=id2word,
                                                                        num_topics=2,
                                                                        random_state=100,
                                                                        chunksize=100,
                                                                        passes=10,
                                                                        iterations=200,
                                                                        per_word_topics=True)
    # Format topics
    sent_topics_df = format_topics_sentences(latent_dirichlet_allocation_model, corpus, data_words)
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Per_Contrib', 'Keywords', 'Text']
    print(df_dominant_topic.head(10))
    print(df_dominant_topic['Dominant_Topic'].unique())
    plt_latent_dirichlet_allocation(latent_dirichlet_allocation_model)
    return latent_dirichlet_allocation_model, df_dominant_topic


def format_topics_sentences(latent_dirichlet_allocation_model, corpus, texts):
    sent_topics_df = pd.DataFrame()
    for i, row_list in enumerate(latent_dirichlet_allocation_model[corpus]):
        row = row_list[0] if latent_dirichlet_allocation_model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = latent_dirichlet_allocation_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Per_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


def plt_latent_dirichlet_allocation(latent_dirichlet_allocation_model):
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = latent_dirichlet_allocation_model.show_topics(formatted=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex="all", sharey="all")

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig('latent_dirichlet_allocation_1.pdf')
    plt.show()

    topics = latent_dirichlet_allocation_model.show_topics(formatted=False)
    data_flat = [w for w_list in data_words for w in w_list]
    counter = Counter(data_flat)
    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    fig, axes = plt.subplots(1, 2, sharey="all")
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030)
        ax.set_ylim(0, 3500)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
    fig.tight_layout(w_pad=2)
    plt.savefig('latent_dirichlet_allocation_2.pdf')
    plt.show()


def principal_components_analysis(documents):
    tfidf_vectorization = TfidfVectorizer(max_features=4500, min_df=15, max_df=0.7, stop_words=stop_words)
    X = tfidf_vectorization.fit_transform(documents)
    x_upd = pd.DataFrame(X.todense(), columns=tfidf_vectorization.get_feature_names())
    scale = StandardScaler()
    X_scaled = scale.fit_transform(x_upd)
    X_normalized = normalize(X_scaled)
    # Converting the numpy array into a pandas DataFrame
    X_normalized = pd.DataFrame(X_normalized)
    pca = PCA(n_components=2)
    X_principal2 = pca.fit_transform(X_normalized)
    X_principal2 = pd.DataFrame(X_principal2)
    X_principal2.columns = ['P1', 'P2']
    print(X_principal2.head())
    plt.scatter(X_principal2.iloc[:, 0], X_principal2.iloc[:, 1], c=y)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig('principal_components_analysis.pdf')
    plt.show()
    return X_scaled


def stochastic_neighbour_embedding(latent_dirichlet_allocation_model, corpus):
    topic_weights = []
    for i, row_list in enumerate(latent_dirichlet_allocation_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])
    arr = pd.DataFrame(topic_weights).fillna(0).values
    arr = arr[np.amax(arr, axis=1) > 0.35]
    topic_num = np.argmax(arr, axis=1)
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)
    n_topics = 2
    colors = np.array([color for color in cols])
    output_file("stochastic_neighbour_embedding.html", title="line plot example")
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=500)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=colors[topic_num])
    show(plot)


def multi_dimensional_scaling():
    topic_binary = np.random.randint(2, size=(100, 10))
    dis_matrix = pairwise_distances(topic_binary, metric='jaccard')
    mds_model = manifold.MDS(n_components=2, random_state=123,
                             dissimilarity='precomputed')
    mds_coords = mds_model.fit_transform(dis_matrix)

    topic_names = ['fill', 'dataset', 'datatable', 'linq', 'query', 'resultset', 'page', 'collection']
    plt.figure()
    plt.scatter(mds_coords[:, 0], mds_coords[:, 1],
                facecolors='none', edgecolors='none')  # points in white (invisible)
    labels = topic_names
    for label, x, y in zip(labels, mds_coords[:, 0], mds_coords[:, 1]):
        plt.annotate(label, (x, y), xycoords='data')
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')
    plt.title('Dissimilarity among Topic')
    plt.savefig('multi_dimensional_scaling.pdf')
    plt.show()


def linear_discriminant_analysis(X_scaled):
    X = X_scaled
    Y = df_dominant_topic['Dominant_Topic'].values
    linear_discriminant_analysis_model = LinearDiscriminantAnalysis()
    X_lda = linear_discriminant_analysis_model.fit_transform(X, Y)
    plt.xlabel('LDA1')
    plt.ylabel('LDA2')
    plt.scatter(
        x=X_lda[:], y=Y
    )
    plt.savefig('linear_discriminant_analysis.pdf')
    plt.show()


def dmm(df_dominant_topic):
    corpus_dm = df_dominant_topic['Text']

    data_dmm = GSDMM.DMM(corpus_dm, 2)

    data_dmm.topicAssigmentInitialise()
    data_dmm.inference()

    data_dmm.worddist()

    finalAssignments = data_dmm.writeTopicAssignments()

    coherence_top_words = data_dmm.writeTopTopicalWords(finalAssignments)

    data_dmm.coherence(coherence_top_words, len(finalAssignments))

    print("Final number of topics found: " + str(len(finalAssignments)))


if __name__ == '__main__':
    file_data = pd.read_excel("title_stack.xlsx")
    file_data['formatted_title'] = file_data['title'].apply(clean_text)
    data = file_data.formatted_title.values.tolist()
    data_words = list(sent_to_words(data))
    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]
    latent_dirichlet_allocation_model, df_dominant_topic = latent_dirichlet_allocation(corpus, id2word, data_words)
    documents = file_data['formatted_title']
    y = df_dominant_topic['Dominant_Topic'].values
    X_scaled = principal_components_analysis(documents)
    stochastic_neighbour_embedding(latent_dirichlet_allocation_model, corpus)
    multi_dimensional_scaling()
    linear_discriminant_analysis(X_scaled)
    dmm(df_dominant_topic)
