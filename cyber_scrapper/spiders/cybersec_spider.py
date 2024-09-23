import os
import scrapy
import psycopg2
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import requests
from dotenv import load_dotenv

load_dotenv()

class CybersecSpider(scrapy.Spider):
    name = "cybersec"

    def __init__(self, query=None, relevancy_threshold=0.75, max_depth=2, *args, **kwargs):
        super(CybersecSpider, self).__init__(*args, **kwargs)
        self.query = query
        self.serpapi_key = os.environ["SERP_KEY"]  # Add your SerpAPI key here
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Pretrained embedding model
        self.query_embedding = self.model.encode(query)
        self.relevancy_threshold = float(relevancy_threshold)  # Set relevancy threshold
        self.max_depth = int(max_depth)  # Set maximum depth for recursive link checking

        # Initialize PostgreSQL connection
        self.conn = psycopg2.connect(
            dbname=os.environ["DB_NAME"],
            user=os.environ["USER_NAME"],
            password=os.environ["PASS"],
            host="localhost",
            port="5432"
        )
        self.cursor = self.conn.cursor()

    def start_requests(self):
        """
        Calls SerpAPI to retrieve Google search results.
        """
        search_url = f"http://api.serpstack.com/search"
        params = {
            "query": self.query,
            "access_key": self.serpapi_key,
            "num": 10,  # Number of results
            "hl": "en"  # Language
        }

        response = requests.get(search_url, params=params)
        search_results = response.json().get('organic_results', [])

        # Loop through each result and initiate a Scrapy request with depth=1
        for result in search_results:
            link = result['url']
            yield scrapy.Request(url=link, callback=self.parse_article, meta={'depth': 1})

    def parse_article(self, response):
        # Get the raw HTML content
        page_content = response.text

        # Use BeautifulSoup to extract the body content
        body_content = self.extract_body_content(page_content)

        # Extract the title of the article
        title = self.extract_title(page_content)

        # Process the body content to extract relevant information
        extracted_content = self.extract_relevant_content(body_content)

        # Generate embeddings for the article content
        article_embedding = self.generate_embedding(extracted_content)

        # Calculate cosine similarity to get relevancy score
        relevancy_score = 1 - cosine(self.query_embedding, article_embedding)

        # Only store the article if relevancy score is above the threshold
        if relevancy_score >= self.relevancy_threshold:
            self.store_article_in_postgresql(title, response.url, extracted_content, article_embedding, relevancy_score)

        # If depth limit is not reached, recursively follow internal links
        current_depth = response.meta['depth']
        if current_depth < self.max_depth and relevancy_score >= self.relevancy_threshold:
            internal_links = self.extract_internal_links(response)
            for link in internal_links:
                # Recursively crawl the internal links with incremented depth
                yield scrapy.Request(url=link, callback=self.parse_article, meta={'depth': current_depth + 1})

    def extract_internal_links(self, response):
        """
        Extracts internal links from the current page.
        """
        internal_links = []
        base_url = response.url
        for link in response.css('a::attr(href)').getall():
            # Check if the link is an internal link (relative or same domain)
            if link.startswith('/') or base_url in link:
                full_link = response.urljoin(link)
                internal_links.append(full_link)
        return internal_links

    def extract_body_content(self, html_content):
        """
        Uses BeautifulSoup to extract content within the <body> tag and remove unnecessary tags.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        body = soup.body

        # Clean the body by removing unnecessary tags
        for script in body(["script", "style", "noscript", "iframe"]):
            script.extract()  # Remove tags like <script>, <style>, etc.

        return ' '.join(body.stripped_strings)  # Return cleaned text content

    def extract_title(self, html_content):
        """
        Extracts the title of the page using BeautifulSoup.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        title = soup.title.string if soup.title else 'No title'
        return title

    def extract_relevant_content(self, body_content):
        """
        Implement your logic to extract relevant information from the body content.
        This can be done using keyword matching, regex, or any other parsing technique.
        """
        # For example, you can simply return the body content here.
        return body_content  # Modify this to include any specific extraction logic

    def generate_embedding(self, text):
        """
        Generates a vector embedding for the given text using SentenceTransformer.
        """
        return self.model.encode(text)

    def store_article_in_postgresql(self, title, url, content, embedding, relevancy_score):
        """
        Stores the article's title, URL, content, embedding, and relevancy score in PostgreSQL if it's relevant.
        """
        # Convert embedding to a list for PostgreSQL
        embedding_list = embedding.tolist()
        
        # Insert the article and its metadata into the PostgreSQL database
        self.cursor.execute("""
            INSERT INTO articles (title, url, content, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING;
        """, (title, url, content, embedding_list))
        self.conn.commit()

    def close(self, reason):
        """
        Close the PostgreSQL connection after the spider finishes.
        """
        self.cursor.close()
        self.conn.close()
