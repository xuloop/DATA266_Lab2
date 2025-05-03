# DATA266_Lab2
Part 1	

MultiModal Retrieval-Augmented Generation

We propose two different approaches for building a MultiModal Retrieval-Augmented Generation (RAG) system that processes a dataset of economic reports, tables, and images to retrieve relevant data and generate meaningful responses to economic queries. The first architecture integrates PyPDF2, SentenceTransformer, scikit-learn's TF-IDF, ChromaDB vector database, and OpenAI's API to create a hybrid retrieval system that employs domain-specific query expansion and context-aware reranking for economic document analysis. The second method uses similar technologies, but employs LangChain's framework to create a comprehensive pipeline for processing PDF documents containing both textual and visual economical information.

The Embedding Pipeline

The embedding pipeline uses the SentenceTransformer library to generate dense vector representations of text, with a preference for the 'intfloat/e5-large-v2' model from Hugging Face. This model has demonstrated superior performance in semantic retrieval tasks compared to earlier models. The e5-large-v2 model, with approximately 335 million parameters, produces 1024-dimensional embeddings that effectively capture semantic relationships between economic concepts, effective on different terminology.

The embedding process is applied differently to various content types - text chunks receive straightforward embedding, while figures and tables undergo a specialized embedding approach that combines their captions with inferred descriptions. For example, a table is embedded not just with its caption but with an expanded description to enhance retrieval relevance.

To complement these dense vectors, the system also employs TF-IDF (Term Frequency-Inverse Document Frequency) vectorization from scikit-learn. This creates sparse vectors that excel at capturing exact matches and rare economic terminology that might get diluted in dense embeddings. We apply TF-IDF with a maximum of 512 features, focusing on the most discriminative terms in the corpus. This hybrid embedding approach combines the semantic understanding of transformers with the precision of TF-IDF—creates a more robust representation space particularly suited to economic documents where both conceptual understanding and terminological precision matter greatly.


The Retrieval Architecture

The retrieval architecture in PrecisionRAG demonstrates a multi-stage approach optimized for economic question answering. The process begins with query expansion, where the original query is enriched with domain-specific terminology based on question classification. For instance, queries about financial crises are expanded with terms like "housing market" and "subprime mortgage," while queries about unemployment incorporate terms like "jobless rate" and "economic hardship." This expansion is informed by domain expertise in economics and significantly improves recall by bridging vocabulary gaps.


The expanded query then undergoes a hybrid retrieval process, querying both the Chroma vector database (using dense embeddings) and the TF-IDF sparse representations. Results from both methods are combined with a preference for dense results, but unique sparse results are preserved to ensure broad coverage. This hybrid approach helps capture both semantic relationships and exact terminology matches.


Retrieved results then undergo a multi-dimensional scoring adjustment based on several factors. Question-specific keywords receive boosting, with matches to terms like "financial crisis" or "unemployment rate" amplifying scores for relevant questions. Content containing numerical data receives an additional boost—recognizing that in economics, statistical evidence is particularly valuable. The system also considers source types, preferentially ranking figures for visually-oriented questions and tables for data-heavy inquiries.


The architecture also includes specialized mechanisms for figure and table retrieval. The code maintains a mapping between question types and relevant visualizations, recognizing that economic questions often have characteristic visual representations. For instance, unemployment questions typically benefit from unemployment rate charts, while GDP-related questions may need tables showing growth figures. When direct mapping isn't available, the system dynamically analyzes content references to identify the most relevant visual elements.
This multi-stage retrieval architecture—query expansion, hybrid retrieval, context-aware reranking, and specialized visual element selection—creates a retrieval system particularly attuned to the complexities of economic documents. 

The BERTScore result of 0.61794 for the RAG system indicates moderate success in generating answers that align semantically with reference responses in the economic domain. BERTScore measures semantic similarity using contextual embeddings, making it more nuanced than exact match metrics like BLEU or ROUGE. This middle-range score is likely reflective of the system's hybrid approach - the neural embeddings successfully capture broad semantic relationships, while the TF-IDF components help with economic terminology.

LangChain Framework

We also implemented a multimodal RAG system specifically designed for economic document analysis, leveraging LangChain's framework to create a comprehensive pipeline for processing PDF documents containing both textual and visual information. The implementation uses PyPDFLoader for text extraction and includes alternate methods for image extraction.


The system employs a content processing approach that generates summaries for both textual and visual elements. For text, it uses GPT-4o to create concise summaries optimized for retrieval, while for images, it uses the same model with vision capabilities to generate descriptive summaries of charts, graphs, and other visual elements commonly found in economic documents. These summaries serve as the basis for vector embeddings created using OpenAI's embedding models.


For knowledge storage and retrieval, the system implements a MultiVectorRetriever backed by ChromaDB. This approach separates the storage of summary embeddings from the original content, allowing the system to retrieve relevant documents based on semantic similarity of summaries while returning the original, unmodified content. The retrieval mechanism handles both text and images, determining which modality is most relevant to a given query.
The RAG chain implementation demonstrates prompt engineering, positioning GPT-4o as a "distinguished macroeconomist" to enhance domain-specific responses. The chain dynamically constructs multimodal prompts that include both retrieved text and relevant images in base64 format, enabling the model to interpret charts and graphs when answering economic questions. 


This implementation differs by using numerous helper functions for tasks like base64 encoding/decoding, image resizing, and type detection, creating a robust system capable of handling the complexities of mixed-modality content typical in economic literature and educational materials.

Conclusion

Comparing the two models, the Langchain RAG implementation achieved a higher BERTScore of 0.69341 compared to the previous RAG's 0.61794. This performance difference likely stems from distinct technical approaches: The previous RAG relies on hybrid embeddings combining SentenceTransformer dense vectors with TF-IDF sparse representations specifically optimized for economic text, while the multimodal Langchain system leverages OpenAI's embedding models and GPT-4o's vision capabilities to process both textual content and visual elements like charts and graphs. The Langchain approach's ability to interpret visual economic data alongside text appears to provide a meaningful advantage in addressing economic questions that often depend on data visualization for complete understanding.

.
Part 2	

Fine-Tuning Stable Diffusion for Image Generation

Part 2 implements a fine-tuning pipeline for Stable Diffusion focused on generating high-quality food images. The pipeline uses the Food101 dataset with LoRA (Low-Rank Adaptation) to efficiently adapt the model while preserving the base model's capabilities. The implementation firstly creates a custom EnhancedFood101Dataset class that loads the Food101 dataset, creates varied text prompts for each food category, processes images to the required resolution, and tokenized text prompts for the model. 

For fine-tuning, we use LoRA adaptation which targets specific modules in both UNet and text encoder components with a configuration of rank=16, alpha=16, and dropout=0.05. The VAE component is frozen to focus training on the text-to-image mapping rather than general image compression.



The training process utilizes the AdamW optimizer with a learning rate of 5e-5, running for 5 epochs with a deliberate noise addition and prediction process in latent space. The base model chosen is "runwayml/stable-diffusion-v1-5" as it offers a good balance between quality and computational requirements. LoRA fine-tuning was selected because it dramatically reduces parameter count (training approximately 1% of parameters), preserves general capabilities while adapting to the food domain, enables faster training, and requires less storage for the resulting model.



For evaluation, CLIP Score was used to measure alignment between generated images and their prompts and Inception Score that assesses diversity and quality of generated images using a pretrained Inception v3 model. 
The image generation uses diverse prompt engineering with 10 different style variations and varying generation parameters (guidance scales 7.0-9.0, step counts 40-50, different random seeds) to maximize diversity. It uses the optimized DPM++ Karras scheduler, to produce higher quality results than default schedulers and better maintains fine details in complex food textures.



The Food101 dataset was chosen because it's a standard benchmark containing 101 food categories with 1,000 images per category, offering real-world food images with natural variation. The code saves fine-tuned model weights, LoRA configuration files, combined checkpoints with timestamps, generated sample images, evaluation metrics, and visualization plots. 


The early rapid loss convergence followed by occasional spikes is common in diffusion models, especially when using LoRA which can adapt rapidly to specific data patterns. The volatility in the later epochs could indicate that the model was encountering highly varied images in the Food101. This is actually a good sign, as it shows the model wasn't settling into a local minimum but continuing to learn from diverse examples and is typical in diffusion model training, as the loss depends greatly on the specific noise levels and images in each batch. The final loss of 0.303169 is relatively high compared to the minimum observed loss of 0.002988, which could mean the learning process was still ongoing and might have benefited from additional epochs.


The CLIP similarity score of 32.3829 indicates excellent alignment between generated images and their text prompts. For context, this score is slightly higher than the dataset (32.1390), suggesting the model generates images that match their text descriptions as well as or better. The dataset reference point is generated using a paired index from the food101 dataset itself, to provide context and standard to where the CLIP scores should be. The substantial gap between this score and the unrelated-content score (11.0407) confirms the model is creating images specifically referencing their respective prompts rather than generic food imagery.
The Inception Score of 7.0800 ± 1.8223 demonstrates good diversity and quality in the generated images. This metric measures both the clarity/quality of individual images and the diversity across the 50 generated images. The score indicates the model is producing varied, distinct food images rather than repetitive or similar outputs. The standard deviation (±1.8223) suggests some variability in generation quality, which is expected when creating diverse food categories.
Part 3	

Developing an Agentic AI Travel Assistant

The purpose of this part is to build up an Agentic AI which is known as the multi agent AI planner including flights, hotel and weather management. By using CrewAI and AI model, we aim to create a flight AI agent, weather AI agent and hotel AI agent separately to retrieve data from APIs platforms and create a centralized workflow to communicate these agents to produce an experienced and personalized travel itinerary based on user queries. 


In short, there is a structure to perform the four agents:


Agentic-ai-travel/
│
├── modules/
│   ├── flight_main.py         # flight_agent + flight_task
│   ├── hotel_main.py          # hotel_agent + hotel_task
│   ├── weather_main.py        # weather_agent + weather_task
│   └── planner_agent_main.py        # planner_agent + planner_task (coordinates others)
│
├── tools/
│   ├── flight_tool.py           # API interaction logic for flights
│   ├── hotel_tool.py            # API interaction logic for hotels
│   └── weather_tool.py          # API interaction logic for weather
│
├── configs/
│   └── amadues_api.py              # API keys 
│
└── README.md                    # Project overview and setup


Flight API Agent:


This agent will get the data from Amadeus API which is known as the API free platform to connect developers with the flight offer search API over 400 airlines, including detailed information such as airlines, departure, arrival, duration time and pricing. To connect with Amadeus API, I create an authorization like client ID and client secret to generate the access token via the url: https://test.api.amadeus.com/v1/security/oauth2/token. The access token is only valid for 30 minutes, so it needs to request a new one periodically. 
The primary function of the Flight Tool is to search for flights and retrieve comprehensive flight information based on user input. It accepts parameters such as origin, destination, departure date, return date, number of travelers, and whether the flight should be non-stop. These parameters are passed to url “https://test.api.amadeus.com/v2/shopping/flight-offers” to retrieve flight information. The tool queries the Amadeus API and returns a curated list of available flight options that match the specified travel criteria.
In the flight main file, I define the flight agent by calling the search flights tool by using crewAI agent. CrewAI is a framework used to facilitate collaboration between AI agents, allowing them to work together on complex tasks like content generation, research, and even automating workflows. It provides a structure for defining agents, assigning tasks, and orchestrating their interactions, essentially enabling AI to operate in a "crew" or team. CrewAI integrates with the AI model to allow each agent to think, plan, and execute tasks based on its role, tools, and goals. Without specific LLM defining, crewAI is going to use gpt 3.5 turbo as the default generation model. 


To test the flight agent, I tried the query “Find the 2 nonstop flights from NYC to LON for 2 people, departing on 2025-05-05 and returning on 2025-05-10.” and the expected output “List 3 good flight options and recommend the best one.”. The AI agent is expected not only to retrieve relevant flight options from the Amadeus API but also to evaluate and recommend the most suitable flight for the user based on the query criteria. 


Weather API Agent:


Weather data is obtained from OpenWeatherMap API which is an online service and provides a current real-time weather and forecast. By using a free API service, there is limited access to retrieve weather forecasts within the next 4 days from today. It restricts selecting a certain or longer date range. 


To access data from this API platform, an HTTP request is made by passing parameters into the URL. The required inputs include the API key, city name or location code, and the current date. The following URL can be used to retrieve temperature and weather conditions:
http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric. The get weather forecast tool will fetch weather data from the next 4 days from the given location and it will be called into the weather agent as the main task.
					
CrewAI acts as the weather agent, taking on the role of a weather assistant with the goal of forecasting weather conditions in a specified location for the next four days. The agent's backstory establishes it as a reliable source for planning weather-dependent activities. A specific task was assigned to this agent: retrieve the weather forecast for London and assist in identifying the best day for outdoor plans.


Hotel API Agent:


Using the same Amadeus API as the Flight agent, the Hotel agent can explore over 150,000 accommodations worldwide. This API provides comprehensive information on hotel pricing, amenities, and user reviews based on the selected dates and location. Since both agents share the same platform, the Hotel agent can reuse the existing authentication method and access tokens, ensuring seamless integration and efficient API communication.


There are the following steps to retrieve hotel information:
The process begins by retrieving a list of hotel IDs for a given city using the endpoint: https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city, where the city code is passed as a parameter. 
These hotel IDs are then used to fetch available hotel offers via the endpoint: https://test.api.amadeus.com/v3/shopping/hotel-offers, along with parameters such as check-in date, check-out date, and the number of adults. 
Finally, user reviews and ratings for each hotel are retrieved using: https://test.api.amadeus.com/v2/e-reputation/hotel-sentiments. By combining data from these three endpoints, the system can present comprehensive hotel details including the hotel name, location, price, room type, description, and guest feedback.
CrewAI will define hotel agent as the role of hotel travel assistant with the goal is to help users find and compare hotel offers with ratings in a given city and date range. This hotel agent uses the tool ‘get_hotel_data_combined’ to perform the specific task like "Search for hotels in the city code 'LON' from 2025-05-05 to 2025-05-10 for 2 adults. Display name, location, price, room type, and rating.". The expectation output is set to list 3 good hotels and recommend the best one based on ratings, prices, and room descriptions.

Planner API Agent:


After developing three functional AI systems, it is required to build a central workflow to facilitate the communication between these agents to deliver a good itinerary plan for travel. Therefore, we create a planner travel agent to manage the plan by reusing the tools from these agents like search flights, get weather data and search hotels. 


The main planner agent is defined as the travel planner assistant, the goal is to create a full travel plan with flights, hotels, and weather forecast. This will call tools: search_flights from flight tool, get_hotel_data_combined from hotel tool and get_weather_data from weather tool. With the user query: 
Plan a complete trip from SFO to PAR from 2025-05-05 to 2025-05-10 for 2 people.
        Use tools to find:
        1.  Flight options.
        2.  Best hotel options.
        3.  Weather forecasts.
An expected plan is a full travel plan itinerary with recommendations for the best 2 flights, hotels, and weather.

CrewAI will start the process to generate the travel plan from flights, then hotel and weather based on the tools provided. The parameters are automatically detected from the query like origin (San Francisco), destination (Paris), departure date/ check-in (2025-05-05), return date/ check-out (2025-05-10), 2 adults. Then it combines these information together as the final result and the AI model suggests the option based on the plan.

