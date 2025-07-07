from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os

# Create directory if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define styles
styles = getSampleStyleSheet()
title_style = styles['Heading1']
title_style.alignment = 1  # Center
heading_style = styles['Heading2']
normal_style = styles['Normal']
normal_style.spaceAfter = 12

# Define content for different PDFs
pdf_content = {
    "intro_to_ml": {
        "title": "Introduction to Machine Learning",
        "content": [
            ("Heading", "What is Machine Learning?"),
            ("Text", "Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. The primary aim is to allow computers to learn automatically without human intervention or assistance and adjust actions accordingly."),
            ("Text", "Machine learning algorithms are trained on data - the more data, the better they perform. The algorithms build mathematical models based on sample data, known as 'training data', in order to make predictions or decisions without being explicitly programmed to perform the task."),
            ("Heading", "Types of Machine Learning"),
            ("Text", "1. Supervised Learning: The algorithm is trained on a labeled dataset, which means each training example is paired with an output label. The goal is to learn a mapping from inputs to outputs."),
            ("Text", "2. Unsupervised Learning: The algorithm is trained on an unlabeled dataset. The goal is to find patterns or structures in the input data."),
            ("Text", "3. Reinforcement Learning: The algorithm learns through trial and error, taking actions to maximize a reward."),
            ("Heading", "Common Machine Learning Algorithms"),
            ("Text", "- Linear Regression: Used for predicting continuous values"),
            ("Text", "- Logistic Regression: Used for binary classification problems"),
            ("Text", "- Decision Trees: Tree-like model of decisions"),
            ("Text", "- Random Forests: Ensemble of decision trees"),
            ("Text", "- Support Vector Machines (SVM): Used for classification and regression"),
            ("Text", "- K-Nearest Neighbors (KNN): Classification based on closest training examples"),
            ("Text", "- Neural Networks: Models inspired by the human brain"),
            ("Heading", "Machine Learning Applications"),
            ("Text", "- Image and Speech Recognition"),
            ("Text", "- Natural Language Processing"),
            ("Text", "- Recommendation Systems"),
            ("Text", "- Autonomous Vehicles"),
            ("Text", "- Fraud Detection"),
            ("Text", "- Healthcare Diagnostics"),
        ]
    },
    "neural_networks": {
        "title": "Neural Networks Explained",
        "content": [
            ("Heading", "What are Neural Networks?"),
            ("Text", "Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way biological neurons signal to one another."),
            ("Text", "Neural networks consist of node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold."),
            ("Heading", "Components of Neural Networks"),
            ("Text", "1. Neurons: Basic units that take inputs, apply weights, and output a value through an activation function."),
            ("Text", "2. Weights: Parameters within a neural network that transform input data within the network's layers."),
            ("Text", "3. Bias: An additional parameter that allows adjustment of the output along with the weighted sum of the inputs to the neuron."),
            ("Text", "4. Activation Function: Determines whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it."),
            ("Heading", "Types of Neural Networks"),
            ("Text", "- Feedforward Neural Networks: The simplest type where information moves in only one direction."),
            ("Text", "- Convolutional Neural Networks (CNNs): Especially useful for image processing and computer vision."),
            ("Text", "- Recurrent Neural Networks (RNNs): Have connections that form cycles, allowing the network to maintain a memory of previous inputs."),
            ("Text", "- Long Short-Term Memory Networks (LSTMs): A special kind of RNN capable of learning long-term dependencies."),
            ("Text", "- Generative Adversarial Networks (GANs): Consist of two neural networks contesting with each other in a zero-sum game framework."),
            ("Heading", "Training Neural Networks"),
            ("Text", "Neural networks learn by adjusting the weights and biases based on the error between their predictions and the actual outputs. This process is known as backpropagation, which is a supervised learning technique used by neural networks."),
            ("Text", "The training process involves:"),
            ("Text", "1. Forward propagation: Input data is fed through the network to get an output."),
            ("Text", "2. Calculating loss: The difference between predicted and actual output."),
            ("Text", "3. Backward propagation: Gradients are calculated and weights are updated to minimize the loss."),
            ("Text", "4. Iteration: Steps 1-3 are repeated until the network performs well enough."),
        ]
    },
    "deep_learning": {
        "title": "Deep Learning Fundamentals",
        "content": [
            ("Heading", "What is Deep Learning?"),
            ("Text", "Deep Learning is a subfield of machine learning that focuses on algorithms inspired by the structure and function of the brain called artificial neural networks. The term 'deep' refers to the number of layers through which the data is transformed."),
            ("Text", "Deep learning excels in learning from large amounts of labeled data and extracting high-level features from raw data automatically, without the need for manual feature extraction."),
            ("Heading", "Why Deep Learning?"),
            ("Text", "Deep learning has revolutionized artificial intelligence by achieving unprecedented accuracy in many important tasks. Some reasons for its success include:"),
            ("Text", "- Ability to automatically learn representations from data"),
            ("Text", "- Excellent performance when trained on large datasets"),
            ("Text", "- Elimination of the need for manual feature engineering"),
            ("Text", "- Versatility across different domains (vision, language, audio, etc.)"),
            ("Heading", "Key Concepts in Deep Learning"),
            ("Text", "1. Feature Hierarchies: Deep learning models learn to represent the world as a nested hierarchy of concepts."),
            ("Text", "2. Representation Learning: The ability to automatically discover useful representations from raw data."),
            ("Text", "3. End-to-End Learning: Training the entire pipeline from raw input to final output in one go."),
            ("Text", "4. Transfer Learning: Reusing a pre-trained model on a new problem."),
            ("Heading", "Popular Deep Learning Architectures"),
            ("Text", "- Deep Neural Networks (DNNs): Multiple hidden layers for complex pattern recognition."),
            ("Text", "- Convolutional Neural Networks (CNNs): Specialized for processing grid-like data such as images."),
            ("Text", "- Recurrent Neural Networks (RNNs): Designed for sequential data processing."),
            ("Text", "- Transformers: Models that use attention mechanisms, revolutionizing NLP."),
            ("Text", "- Autoencoders: Used for unsupervised learning of efficient codings."),
            ("Heading", "Deep Learning Applications"),
            ("Text", "- Computer Vision (object detection, image classification)"),
            ("Text", "- Natural Language Processing (translation, sentiment analysis)"),
            ("Text", "- Speech Recognition"),
            ("Text", "- Autonomous Vehicles"),
            ("Text", "- Medical Diagnosis"),
            ("Text", "- Game Playing (AlphaGo, AlphaZero)"),
        ]
    },
    "nlp_basics": {
        "title": "Natural Language Processing Basics",
        "content": [
            ("Heading", "What is Natural Language Processing?"),
            ("Text", "Natural Language Processing (NLP) is a field of artificial intelligence that gives computers the ability to understand, interpret, and generate human language in a way that is valuable. NLP combines computational linguistics, machine learning, and deep learning models to process human language."),
            ("Text", "NLP enables computers to perform various tasks involving language, from simple ones like spell checking and keyword searches to more complex operations like automatic summarization, translation, sentiment analysis, and question answering."),
            ("Heading", "Key Components of NLP"),
            ("Text", "1. Tokenization: Breaking down text into words, phrases, or other meaningful elements."),
            ("Text", "2. Part-of-Speech Tagging: Identifying parts of speech for each word in a text."),
            ("Text", "3. Named Entity Recognition: Identifying named entities such as persons, organizations, locations, etc."),
            ("Text", "4. Parsing: Analyzing the grammatical structure of sentences."),
            ("Text", "5. Semantic Analysis: Understanding the meaning of text."),
            ("Heading", "NLP Techniques"),
            ("Text", "- Statistical Methods: Using statistical algorithms to understand and generate language."),
            ("Text", "- Machine Learning Methods: Training algorithms on large corpora of text."),
            ("Text", "- Rule-Based Methods: Using handcrafted linguistic rules."),
            ("Text", "- Neural Networks: Employing deep learning techniques for NLP tasks."),
            ("Text", "- Transfer Learning: Using pre-trained language models like BERT, GPT, etc."),
            ("Heading", "NLP Applications"),
            ("Text", "- Machine Translation: Converting text from one language to another."),
            ("Text", "- Sentiment Analysis: Determining the sentiment expressed in a text."),
            ("Text", "- Text Summarization: Creating concise and coherent summaries of longer documents."),
            ("Text", "- Question Answering: Providing answers to questions posed in natural language."),
            ("Text", "- Chatbots and Virtual Assistants: Creating interactive systems that communicate with humans."),
            ("Text", "- Information Extraction: Automatically extracting structured information from unstructured text."),
            ("Heading", "Challenges in NLP"),
            ("Text", "- Ambiguity: Words and sentences can have multiple meanings."),
            ("Text", "- Context: Understanding the context in which language is used."),
            ("Text", "- Cultural Nuances: Grasping cultural references and idioms."),
            ("Text", "- Sarcasm and Irony: Detecting and understanding non-literal language."),
        ]
    },
    "computer_vision": {
        "title": "Computer Vision Fundamentals",
        "content": [
            ("Heading", "What is Computer Vision?"),
            ("Text", "Computer Vision is a field of artificial intelligence that enables computers to derive meaningful information from digital images, videos, and other visual inputs. It involves developing algorithms to accomplish tasks that the human visual system can do, such as recognizing objects, interpreting scenes, and understanding context."),
            ("Text", "The goal of computer vision is to automate tasks that the human visual system can do, making machines capable of identifying and classifying objects and then reacting to what they 'see.'"),
            ("Heading", "Core Concepts in Computer Vision"),
            ("Text", "1. Image Classification: Categorizing what is contained in an image."),
            ("Text", "2. Object Detection: Identifying instances of semantic objects of a certain class in an image."),
            ("Text", "3. Image Segmentation: Dividing an image into segments to simplify or change its representation."),
            ("Text", "4. Feature Extraction: Identifying key features within images for further analysis."),
            ("Text", "5. Pattern Recognition: Identifying patterns and regularities in data."),
            ("Heading", "Computer Vision Techniques"),
            ("Text", "- Traditional Methods: Using filters, edge detection, feature extraction, etc."),
            ("Text", "- Machine Learning: Training algorithms on labeled image datasets."),
            ("Text", "- Deep Learning: Using convolutional neural networks (CNNs) for visual tasks."),
            ("Text", "- Transfer Learning: Utilizing pre-trained models for new computer vision tasks."),
            ("Heading", "Popular Computer Vision Algorithms"),
            ("Text", "- SIFT (Scale-Invariant Feature Transform): For feature detection and description."),
            ("Text", "- HOG (Histogram of Oriented Gradients): For object detection."),
            ("Text", "- YOLO (You Only Look Once): For real-time object detection."),
            ("Text", "- R-CNN (Region-based Convolutional Neural Networks): For object detection."),
            ("Text", "- U-Net: For image segmentation."),
            ("Heading", "Computer Vision Applications"),
            ("Text", "- Facial Recognition: Identifying individuals in images or video."),
            ("Text", "- Autonomous Vehicles: Enabling cars to 'see' and navigate their environment."),
            ("Text", "- Medical Imaging: Analyzing X-rays, MRIs, and other medical images for diagnosis."),
            ("Text", "- Surveillance: Monitoring for security purposes."),
            ("Text", "- Augmented Reality: Overlaying digital information on the real world."),
            ("Text", "- Manufacturing: Quality control and defect detection."),
        ]
    }
}

def create_pdf(title, content, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    
    # Add title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.5 * inch))
    
    # Add content
    for item_type, item_content in content:
        if item_type == "Heading":
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(item_content, heading_style))
            story.append(Spacer(1, 0.1 * inch))
        elif item_type == "Text":
            story.append(Paragraph(item_content, normal_style))
    
    doc.build(story)
    print(f"Created PDF: {filename}")

# Generate PDFs
for key, data in pdf_content.items():
    output_filename = os.path.join(current_dir, f"{key}.pdf")
    create_pdf(data["title"], data["content"], output_filename)

print("All PDFs have been generated successfully!")
