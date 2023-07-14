#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyaspeller import YandexSpeller
from nltk.corpus import wordnet

# Download WordNet resource
nltk.download('wordnet')
nltk.download('punkt')
# Keyword Analysis
def suggest_keywords(blog_content, target_audience):
    # Tokenize the blog content into individual words
    tokens = word_tokenize(blog_content)

    # Remove stop words and punctuation
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    # Join the filtered tokens into a string
    filtered_text = " ".join(filtered_tokens)

    # Vectorize the filtered text using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([filtered_text])

    # Calculate the cosine similarity between the TF-IDF matrix and the target audience
    target_vector = vectorizer.transform([target_audience])
    cosine_similarities = cosine_similarity(tfidf_matrix, target_vector)

    # Get the top 5 most similar words to the target audience
    most_similar_indices = cosine_similarities.flatten().argsort()[-5:][::-1]
    suggested_keywords = [vectorizer.get_feature_names_out()[index] for index in most_similar_indices]

    return suggested_keywords


# On-Page Optimization
def optimize_onpage_elements(blog_content):
    optimization_recommendations = {}

    # Optimize meta tags
    meta_title = get_meta_title(blog_content)
    meta_description = get_meta_description(blog_content)
    optimization_recommendations['Meta Title'] = optimize_meta_title(meta_title)
    optimization_recommendations['Meta Description'] = optimize_meta_description(meta_description)

    # Optimize headings
    headings = get_headings(blog_content)
    optimization_recommendations['Headings'] = optimize_headings(headings)

    # Optimize URL structure
    url = get_url(blog_content)
    optimization_recommendations['URL'] = optimize_url(url)

    # Optimize image tags
    images = get_images(blog_content)
    optimization_recommendations['Images'] = optimize_images(images)

    # Optimize internal linking
    internal_links = get_internal_links(blog_content)
    optimization_recommendations['Internal Links'] = optimize_internal_links(internal_links)

    return optimization_recommendations


# Content Analysis
def analyze_content(blog_content):
    readability_score = calculate_readability_score(blog_content)
    keyword_density = calculate_keyword_density(blog_content)
    content_quality = assess_content_quality(blog_content)

    return readability_score, keyword_density, content_quality


# Helper functions
def get_meta_title(blog_content):
    # Extract meta title from the blog content
    meta_title = ""
    # Code to extract meta title
    return meta_title


def optimize_meta_title(meta_title):
    # Optimize the meta title and provide recommendations
    optimization_recommendations = {}
    # Code to optimize meta title
    return optimization_recommendations


def get_meta_description(blog_content):
    # Extract meta description from the blog content
    meta_description = ""
    # Code to extract meta description
    return meta_description


def optimize_meta_description(meta_description):
    # Optimize the meta description and provide recommendations
    optimization_recommendations = {}
    # Code to optimize meta description
    return optimization_recommendations


def get_headings(blog_content):
    # Extract headings from the blog content
    headings = []
    # Code to extract headings
    return headings


def optimize_headings(headings):
    # Optimize the headings and provide recommendations
    optimization_recommendations = {}
    # Code to optimize headings
    return optimization_recommendations


def get_url(blog_content):
    # Extract URL from the blog content
    url = ""
    # Code to extract URL
    return url


def optimize_url(url):
    # Optimize the URL and provide recommendations
    optimization_recommendations = {}
    # Code to optimize URL
    return optimization_recommendations


def get_images(blog_content):
    # Extract images from the blog content
    images = []
    # Code to extract images
    return images


def optimize_images(images):
    # Optimize the image tags and provide recommendations
    optimization_recommendations = {}
    # Code to optimize image tags
    return optimization_recommendations


def get_internal_links(blog_content):
    # Extract internal links from the blog content
    internal_links = []
    # Code to extract internal links
    return internal_links


def optimize_internal_links(internal_links):
    # Optimize the internal links and provide recommendations
    optimization_recommendations = {}
    # Code to optimize internal links
    return optimization_recommendations


def calculate_readability_score(blog_content):
    words = nltk.word_tokenize(blog_content.lower())
    sentences = nltk.sent_tokenize(blog_content)

    # Calculate the average number of words per sentence
    words_per_sentence = len(words) / len(sentences)

    # Calculate the average number of syllables per word
    syllables_per_word = calculate_syllables_per_word(words)

    # Calculate the Flesch-Kincaid readability score
    readability_score = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59

    return readability_score


def calculate_syllables_per_word(words):
    syllables_count = 0
    vowels = 'aeiouy'

    for word in words:
        word_syllables = 0
        word = word.lower()
        if word[0] in vowels:
            word_syllables += 1

        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                word_syllables += 1

        if word.endswith('e'):
            word_syllables -= 1

        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            word_syllables += 1

        if word_syllables == 0:
            word_syllables += 1

        syllables_count += word_syllables

    syllables_per_word = syllables_count / len(words)
    return syllables_per_word


def calculate_keyword_density(blog_content):
    words = nltk.word_tokenize(blog_content.lower())

    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    total_words = len(filtered_words)
    word_counts = nltk.FreqDist(filtered_words)

    target_word = 'keyword'  # Replace with your target keyword
    target_word_count = word_counts[target_word] if target_word in word_counts else 0

    keyword_density = (target_word_count / total_words) * 100
    return keyword_density


def assess_content_quality(blog_content):
    # Perform spell check
    spelling_errors = perform_spell_check(blog_content)

    # Calculate uniqueness score
    words = nltk.word_tokenize(blog_content.lower())
    unique_words = set(words)
    uniqueness_score = len(unique_words) / len(words)

    # Calculate synonym density
    synonym_density = calculate_synonym_density(words)

    # Calculate content quality score
    content_quality = spelling_errors * uniqueness_score * synonym_density

    return content_quality


def perform_spell_check(text):
    spelling_errors = 0

    # Perform spell check using the pyaspeller library
    speller = YandexSpeller()
    result = speller.spell(text)
    for word in result:
        if word["code"] != 1:
            spelling_errors += 1

    return spelling_errors


def calculate_synonym_density(words):
    synonyms_count = 0
    total_words = len(words)

    for word in words:
        synsets = wordnet.synsets(word)
        if len(synsets) > 1:
            synonyms_count += 1

    synonym_density = synonyms_count / total_words
    return synonym_density


def generate_seo_score(blog_content):
    # Define your SEO score weights
    keyword_weight = 0.4
    content_quality_weight = 0.3

    # Calculate the individual SEO scores
    keyword_score = calculate_keyword_density(blog_content)
    content_quality_score = assess_content_quality(blog_content)

    # Calculate the overall SEO score
    seo_score = (
        keyword_weight * keyword_score +
        content_quality_weight * content_quality_score
    )

    return seo_score


def provide_seo_suggestions(blog_content):
    # Perform SEO analysis and provide actionable suggestions
    suggestions = []

    # Keyword Analysis
    keywords = suggest_keywords(blog_content, target_audience="target audience")
    suggestions.append(f"Keywords: {', '.join(keywords)}")

    # On-page Optimization
    onpage_suggestions = optimize_onpage_elements(blog_content)
    suggestions.extend(onpage_suggestions.values())

    # Content Analysis
    readability_score, keyword_density, content_quality = analyze_content(blog_content)
    suggestions.append(f"Readability Score: {readability_score}")
    suggestions.append(f"Keyword Density: {keyword_density}")
    suggestions.append(f"Content Quality: {content_quality}")

    return suggestions


def generate_serp_preview(title, meta_description, url):
    serp_preview = ""

    # Limit the title length to a maximum of 60 characters
    truncated_title = title[:60] if len(title) > 60 else title

    # Limit the meta description length to a maximum of 160 characters
    truncated_meta_description = meta_description[:160] if len(meta_description) > 160 else meta_description

    # Construct the SERP preview snippet
    serp_preview += f"Title: {truncated_title}\n"
    serp_preview += f"Description: {truncated_meta_description}\n"
    serp_preview += f"URL: {url}"

    return serp_preview


def analyze_blog_content(blog_content):
    # Keyword Analysis
    keywords = suggest_keywords(blog_content, target_audience="target audience")

    # Content Analysis
    readability_score, keyword_density, content_quality = analyze_content(blog_content)

    # SEO Score
    seo_score = generate_seo_score(blog_content)

    # SEO Suggestions
    seo_suggestions = provide_seo_suggestions(blog_content)

    # SERP Preview
    title = "The Yugas"  # Replace with the actual blog title
    meta_description = "In Hinduism there are an infinite number of universes which are being created and destroyed. When the Lord Vishnu inhales many universes are destroyed and when he exhales many universes are created. Each of these universes have their own Brahma (creator), Vishnu (preserver) & Shiva (destroyer)."  # Replace with the actual meta description
    url = "https://bhargavchandu.wordpress.com/2023/01/22/the-yugas/"  # Replace with the actual blog URL
    serp_preview = generate_serp_preview(title, meta_description, url)

    # Compile analysis results
    analysis_results = {
        "Keywords": keywords,
        "Readability Score": readability_score,
        "Keyword Density": keyword_density,
        "Content Quality": content_quality,
        "SEO Score": seo_score,
        "SEO Suggestions": seo_suggestions,
        "SERP Preview": serp_preview
    }

    return analysis_results


# Example usage
blog_contents = [
    "In Satya Yuga, the human race was immersed in meditation and possessed spiritual strength and longevity. There was no war, conflict, or famine. Dharma was considered supreme, and people indulged in good karmas. There was complete peace on earth. The human intellect can comprehend all, even God the Spirit beyond this visible world.” This is the age when we will be so spiritually advanced, that God himself is part of our everyday experience. We will be able see God in everyone and everything. Our intuition will be completely developed, and we will live in happiness and natural simplicity. We will understand the nature of the universe, and be able to communicate telepathically to anywhere in the world.Life span of human Beings = 1 lakh yearsPeople can able to fly and create things at will People look young their entire lives and do not get diseases People can communicate with animals and live in harmony with nature and each other The weather is perfect and there is plenty of food Everyone can communicate in one languageIn satya Yuga,Matsya, Kurma, Varaha, Narasimha, Vamana, Parashurama, Rama, Krishna, Dhanvantari, and Kalki are the most popular and widely known 10 avatars of Vishnu( Dasavatara of Vishnu). Out of 10 avatara of Vishnu, in Satya yuga-Matsya, Kurma, Varaha, Narasimha, and Dhanvantari were born in this golden age.",
    "In Treta Yuga, we recognize that everything is interconnected. We will have strong mental powers such as telekinesis, mental telepathy, and the ability to manifest whatever we need. If there is any war, it will not be widespread; most people will be peace-loving and compassionate. Just more spiritually advanced. Being more spiritually advanced and able to manifest at will whatever we need, we may actually leave technology behind and choose to live more simply and naturally. In this silver age Sri Ramachandra, Vamana and Parashuram were born in this yuga to protect dharma.",
    "The Dvapara Yuga, talks about the life and times of Lord Krishna, the ninth avatara of Maha Vishnu. The Dvapara Yuga ended when Krishna completed his mission and returned to his original abode at Vaikuntha. The two highlights of Dvapara Yuga are Kama and Artha. The Vedas were categorised into four parts, Rig, Sama, Yajur and Atharva. The human race began to stray from the righteous path of dharma, much before the beginning of Dvapara Yuga.  People exploited their positions at the expense of others. Wars broke out as kings vied for power, wealth, and influence. However, thousands of temples were constructed around the world during this time. "
]

for blog_content in blog_contents:
    analysis_results = analyze_blog_content(blog_content)
    print("Analysis Results:")
    for key, value in analysis_results.items():
        print(key, ":", value)
    print("-----------------------------------------")

