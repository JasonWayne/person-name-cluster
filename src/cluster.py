# Use John Kennedy's case to develop a basic version
# just use the title and snippet to cluster
import xml.etree.cElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

description_path = '../fixtures/training/web_pages/John_Kennedy/\
John_Kennedy.xml'
web_page_folder_path = '../fixtures/training/web_pages/John_Kennedy'


class PersonNameDoc():
	def __init__(self, root):
		self.search_string = root.attrib['search_string']
		self.doc_list = []
		for child in root:
			rank = child.attrib['rank']
			title = child.attrib['title']
			url = child.attrib['url']
			snippet = child.getchildren()[0].text
			entry = Entry(rank, title, url, snippet)
			self.doc_list.append(entry)

	def __str__(self):
		return self.search_string


class Entry():
	def __init__(self, rank, title, url, snippet):
		self.rank = rank
		self.title = title
		self.url = url
		self.snippet = snippet

	def __str__(self):
		return 'rank : {}, title : {}, url : {}, snippet : {}'.format(
			self.rank, self.title, self.url, self.snippet)


class Cluster():
	def __init__(self, person):
		self.person = person
		self.train = []
		self.vectorizer = self.feature_extraction()
		self.kmeans = self.clustering()
		self.result = self.test()

	def feature_extraction(self):
		vectorizer = CountVectorizer(binary=True, stop_words='english')
		corpus = []
		for doc in self.person.doc_list:
			corpus.append(doc.title + ' ' + doc.snippet)
		self.train = vectorizer.fit_transform(corpus)
		return vectorizer

	def clustering(self):
		kmeans = KMeans(n_clusters=26)
		kmeans.fit_transform(self.train)
		return kmeans

	def test(self):
		d = {}
		for index, line in enumerate(self.train):
			class_type = self.kmeans.predict(line)[0]
			doc_id = self.person.doc_list[index].rank
			if d.get(class_type):
				d[class_type].append(doc_id)
			else:
				d[class_type] = [doc_id]
		return d


def get_data_from_xml(xml_path):
	tree = ET.parse(xml_path)
	root = tree.getroot()
	return root

if __name__ == "__main__":
	cluster = Cluster(PersonNameDoc(get_data_from_xml(description_path)))
