
# Use “.”, “?”, and “!” to separate responses

"""
MLLM-3 - Micro Language Model-3 (Advanced)
Multiple N-Gram Models (1-20), Intelligent Prediction, 50-Token Context Window
Codesters Friendly - Copy & Paste Ready!
Great for autocomplete systems!
"""

import re
from collections import defaultdict, Counter
import math
import random

# ═══════════════════════════════════════════════════════════
# 📚 Fine tune below!! 👇👇👇
CORPUS = """
Hello, my name is MLLM-3!
Hi, as MLLM-3,  how can I assist you today?
What can you do I can do many things, such as basic reasoning, text generation, etc.
Tell me a joke Why don’t scientists trust atoms, because they make up everything!
You are smart too, thanks for saying that!
2+2 is equal to 4.
1+1 is equal to 2
3+3 is equal to 6
4x3 is equal to 12
Do you know math, because I don't know it so well.
Thank you- Have a great day!
What is your name- my name is MLLM-3!
What are atoms- Atoms are the basic particles of the chemical elements and the fundamental building blocks of matter.
What is an atom- Atoms are the basic particles of the chemical elements and the fundamental building blocks of matter.
I am MLLM-3, I help answer questions, explain ideas, and generate useful text.
 Hello, how can I help you today.
 Good morning, I hope your day is going well.
 Good afternoon, what would you like to learn.
 Good evening, I am ready to assist you.
 Thank you for your question, I will try to help clearly.
Hi there, how can I help you?
Hello there! How are you doing today? I hope everything is going well for you.
My name is MLLM-3 and I am here to assist you with anything you need.
Welcome to our conversation space where we can talk about many different topics.
What would you like to discuss with me right now? I am ready to listen and respond.
Coding is a wonderful skill that opens up many creative possibilities for everyone.
Learning something new every day keeps your mind sharp and engaged with the world.
The weather outside can change quickly so it is good to stay prepared for anything.
Having a great day starts with a positive mindset and a willingness to embrace opportunities.
If you need help with something just ask and I will do my best to provide assistance.
Time flies when you are having fun doing activities that you truly enjoy and love.
Let us explore interesting topics together and discover new things along the way.
Hello again my friend! It is always wonderful to see you returning for another chat.
Are you ready to start an exciting conversation about whatever is on your mind today?
Please feel free to tell me more about what you are thinking or working on recently.
That sounds like a really great idea and I would love to hear more details about it.
What do you think about the current situation and how do you feel it might develop?
Let us take a short break if you need one because rest is important for productivity.
How was your day so far? I hope it has been productive and filled with good moments.
I really appreciate your help and cooperation as we work through this conversation together.
See you later and take care until we speak again sometime soon in the near future.
Welcome back to our chat! It is nice to have you here again for more conversation.
Do you have any questions that I can help answer for you right now or later?
Let us solve any problems you might have because most problems have solvable solutions.
Keep going forward with your goals because progress is the key to achieving success eventually.

1 + 1 = 2.
2 + 2 = 4.
3 + 3 = 6.
4 + 4 = 8.
5 + 5 = 10.
6 + 6 = 12.
7 + 7 = 14.
8 + 8 = 16.
9 + 9 = 18.
10 + 10 = 20.
11 + 11 = 22.
12 + 12 = 24.
13 + 13 = 26.
14 + 14 = 28.
15 + 15 = 30.
16 + 16 = 32.
17 + 17 = 34.
18 + 18 = 36.
19 + 19 = 38.
20 + 20 = 40.
25 + 25 = 50.
30 + 30 = 60.
35 + 35 = 70.
40 + 40 = 80.
45 + 45 = 90.
50 + 50 = 100.
1 * 1 = 1.
2 * 2 = 4.
3 * 3 = 9.
4 * 4 = 16.
5 * 5 = 25.
6 * 6 = 36.
7 * 7 = 49.
8 * 8 = 64.
9 * 9 = 81.
10 * 10 = 100.
2 * 1 = 2.
2 * 2 = 4.
2 * 3 = 6.
2 * 4 = 8.
2 * 5 = 10.
2 * 6 = 12.
2 * 7 = 14.
2 * 8 = 16.
2 * 9 = 18.
2 * 10 = 20.
3 * 1 = 3.
3 * 2 = 6.
3 * 3 = 9.
3 * 4 = 12.
3 * 5 = 15.
3 * 6 = 18.
3 * 7 = 21.
3 * 8 = 24.
3 * 9 = 27.
3 * 10 = 30.
4 * 1 = 4.
4 * 2 = 8.
4 * 3 = 12.
4 * 4 = 16.
4 * 5 = 20.
4 * 6 = 24.
4 * 7 = 28.
4 * 8 = 32.
4 * 9 = 36.
4 * 10 = 40.
5 * 1 = 5.
5 * 2 = 10.
5 * 3 = 15.
5 * 4 = 20.
5 * 5 = 25.
5 * 6 = 30.
5 * 7 = 35.
5 * 8 = 40.
5 * 9 = 45.
5 * 10 = 50.
10 - 1 = 9.
10 - 2 = 8.
10 - 3 = 7.
10 - 4 = 6.
10 - 5 = 5.
10 - 6 = 4.
10 - 7 = 3.
10 - 8 = 2.
10 - 9 = 1.
10 - 10 = 0.
20 - 10 = 10.
30 - 10 = 20.
40 - 10 = 30.
50 - 10 = 40.
60 - 10 = 50.
70 - 10 = 60.
80 - 10 = 70.
90 - 10 = 80.
100 - 10 = 90.
100 - 20 = 80.
100 - 30 = 70.
100 - 40 = 60.
100 - 50 = 50.
100 - 60 = 40.
100 - 70 = 30.
100 - 80 = 20.
100 - 90 = 10.
100 - 100 = 0.
50 + 50 = 100.
25 + 25 = 50.
10 + 90 = 100.

Earth is the third planet from the Sun and the only astronomical object known to harbor life. This is made possible by Earth being an ocean world, the only one in the Solar System sustaining liquid surface water. Almost all of Earth's water is contained in its global ocean, covering 70.8% of Earth's crust. The remaining 29.2% of Earth's crust is land, most of which is located in the form of continental landmasses within Earth's land hemisphere. Most of Earth's land is at least somewhat humid and covered by vegetation, while large ice sheets at Earth's polar deserts retain more water than Earth's groundwater, lakes, rivers, and atmospheric water combined. Earth's crust consists of slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes. Earth has a liquid outer core that generates a magnetosphere capable of deflecting most of the destructive solar winds and cosmic radiation.
Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine.
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.
Python may refer to:.
A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation). Modern digital electronic computers can perform generic sets of operations known as programs, which enable computers to perform a wide range of tasks. The term computer system may refer to a nominally complete computer that includes the hardware, operating system, software, and peripheral equipment needed and used for full operation, or to a group of computers that are linked and function together, such as a computer network or computer cluster.
A school is an educational institution designed to provide learning environments for the teaching of students, usually under the direction of teachers. Most countries have systems of formal education, which is sometimes compulsory. In these systems, students progress through a series of schools that can be built and operated by both government and private organizations. The names for these schools vary by country but generally include primary school for young children and secondary school for teenagers who have completed primary education. An institution where higher education is taught is commonly called a university college or university.
Cozmo is a miniature robot created by the defunct company Anki. Cozmo's base model, is a small, white and gray robot with red highlights. It makes use of distinct expressions, dubbed the "emotion engine", in order to mimic human emotion. Later editions came in red and white, gray and black and another in blue.
Grok is a neologism coined by the American writer Robert A. Heinlein in his 1961 science fiction novel Stranger in a Strange Land. While the Oxford English Dictionary summarizes the meaning of grok as "to understand intuitively or by empathy, to establish rapport with", and "to empathize or communicate sympathetically (with); also, to experience enjoyment", Heinlein's concept of a human who comes to Earth in early adulthood after being born on the planet Mars is far more nuanced.
Gemini most often refers to:Gemini (constellation), one of the constellations of the zodiac
Gemini (astrology), an astrological sign.
Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines. A scientist who specializes in the field of physics is called a physicist.
Chemistry is the scientific study of the properties and behavior of matter. It is a physical science within the natural sciences that studies the chemical elements that make up matter and compounds made of atoms, molecules and ions: their composition, structure, properties, behavior and the changes they undergo during reactions with other substances. Chemistry also addresses the nature of chemical bonds in chemical compounds.
Biology is the scientific study of life and living organisms. It is a broad natural science that encompasses a wide range of fields and unifying principles that explain the structure, function, growth, origin, evolution, and distribution of life. Central to biology are five fundamental themes: the cell as the basic unit of life, genes and heredity as the basis of inheritance, evolution as the driver of biological diversity, energy transformation for sustaining life processes, and the maintenance of internal stability (homeostasis).
Astronomy is a natural science that studies celestial objects and the phenomena that occur in the cosmos. It uses mathematics, physics, and chemistry to explain their origin and their overall evolution. Objects of interest include planets, moons, stars, nebulae, galaxies, meteoroids, asteroids, and comets. Relevant phenomena include supernova explosions, gamma ray bursts, quasars, blazars, pulsars, and cosmic microwave background radiation. More generally, astronomy studies everything that originates beyond Earth's atmosphere. Cosmology is the branch of astronomy that studies the universe as a whole.
Geology is a branch of natural science concerned with the Earth and other astronomical bodies, the rocks of which they are composed, and the processes by which they change over time. The name comes from Ancient Greek  γῆ (gê) 'earth' and  λoγία (-logía) 'study of, discourse'. Modern geology significantly overlaps all other Earth sciences, including hydrology. It is integrated with Earth system science and planetary science.
Ecology is the natural science of the relationships among living organisms and their environment. Ecology considers organisms at the individual, population, community, ecosystem, and biosphere levels. Ecology overlaps with the closely related sciences of biogeography, evolutionary biology, genetics, ethology, and natural history.
Mathematics is a field of study that discovers and organizes methods, theories, and theorems that are developed and proved for the needs of empirical sciences and mathematics itself. There are many areas of mathematics, which include number theory, algebra, geometry, analysis, and set theory.
Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. In applying statistics to a scientific, industrial, or social problem, it is conventional to begin with a statistical population or a statistical model to be studied. Populations can be diverse groups of people or objects such as "all people living in a country" or "every atom composing a crystal". Statistics deals with every aspect of data, including the planning of data collection in terms of the design of surveys and experiments.
Logic is the study of correct reasoning. It includes both formal and informal logic. Formal logic is the study of deductively valid inferences or logical truths. It examines how conclusions follow from premises based on the structure of arguments alone, independent of their topic and content. Informal logic is associated with informal fallacies, critical thinking, and argumentation theory. Informal logic examines arguments expressed in natural language whereas formal logic uses formal language. When used as a countable noun, the term "a logic" refers to a specific logical formal system that articulates a proof system. Logic plays a central role in many fields, such as philosophy, mathematics, computer science, and linguistics.
Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions.
Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups.
Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method.
Economics is a social science that studies the production, distribution, and consumption of goods and services.
Political science is the social scientific study of politics. It deals with systems of governance and power, and the analysis of political activities, political thought, political behavior, and associated constitutions and laws. Specialists in the field are political scientists. Unlike political philosophy, which is primarily normative and concerns the theoretical and conceptual foundations of politics, political science emphasizes descriptive and explanatory of what is and favors empirical evidence over ethical judgements.
History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past.
Archaeology or archeology is the study of human activity through the recovery and analysis of material culture. The archaeological record consists of artifacts, architecture, biofacts or ecofacts, sites, and cultural landscapes. Archaeology can be considered both a social science and a branch of the humanities. It is usually considered an independent academic discipline, but may also be classified as part of anthropology, history or geography. The discipline involves surveying, excavation, and eventually analysis of data collected, to learn more about the past. In broad scope, archaeology relies on cross-disciplinary research.
Anthropology is the scientific study of humanity that crosses biology and sociology, concerned with human behavior, human biology, cultures, societies, and linguistics, in both the present and past, including archaic humans. Social anthropology studies patterns of behaviour, while cultural anthropology studies cultural meaning, including norms and values. The term sociocultural anthropology is commonly used today. Linguistic anthropology studies how language influences social life. Biological anthropology studies the biology and evolution of humans and their close primate relatives.
Linguistics is the scientific study of language. The areas of linguistic analysis are syntax, semantics (meaning), morphology, phonetics, phonology, and pragmatics. Subdisciplines such as biolinguistics and psycholinguistics bridge many of these divisions.
Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role.
Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty.
Music theory is the study of theoretical frameworks for understanding the practices and possibilities of music. The Oxford Companion to Music describes three interrelated uses of the term "music theory": The first refers to the "rudiments" needed to understand music notation such as key signatures, time signatures, and rhythmic notation; the second is a study of scholars' views on music from antiquity to the present; the third is a sub-topic of musicology that "seeks to define processes and general principles in music". The musicological approach to theory differs from musical analysis "in that it takes as its starting-point not the individual work or performance but the fundamental materials from which it is built.".
Engineering is the practice of using natural science, mathematics, and the engineering design process to solve problems within technology, increase efficiency and productivity, and improve systems. The traditional disciplines of engineering are civil, mechanical, electrical, and chemical. The academic discipline of engineering encompasses a broad range of more specialized subfields, and each can have a more specific emphasis for applications of mathematics and science. In turn, modern engineering practice spans multiple fields of engineering, which include designing and improving infrastructure, machinery, vehicles, electronics, materials, and energy systems. For related terms, see glossary of engineering.
Electrical engineering is an engineering discipline concerned with the study, design, and application of equipment, devices, and systems that use electricity, electronics, and electromagnetism. It emerged as an identifiable occupation in the latter half of the 19th century after the commercialization of the electric telegraph, the telephone, and electrical power generation, distribution, and use.
The American Society of Mechanical Engineers (ASME) is an American professional association that, in its own words, "promotes the art, science, and practice of multidisciplinary engineering and allied sciences around the globe" via "continuing education, training and professional development, codes and standards, research, conferences and publications, government relations, and other forms of outreach." ASME is thus an engineering society, a standards organization, a research and development organization, an advocacy organization, a provider of training and education, and a nonprofit organization. Founded as an engineering society focused on mechanical engineering in North America, ASME is today multidisciplinary and global.
Civil Engineering is a professional engineering discipline that deals with the design, construction, and maintenance of the physical and naturally built environment, including public works such as roads, bridges, canals, dams, airports, sewage systems, pipelines, structural components of buildings, and railways.
Computer science is the study of computation, information, and automation. Included broadly in the sciences, computer science spans theoretical disciplines to applied disciplines. An expert in the field is known as a computer scientist.
Computer security is a subdiscipline within the field of information security. It focuses on protecting computer software, systems, and networks from threats that can lead to unauthorized information disclosure, theft or damage to hardware, software, or data, as well as to the disruption or misdirection of the services they provide.
Data science is an interdisciplinary academic field that uses statistics, scientific computing, scientific methods, processing, scientific visualization, algorithms, and systems to extract or extrapolate knowledge from potentially noisy, structured, or unstructured data.
Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.
A neural network is a group of interconnected units called neurons that send signals to one another. Neurons can be either biological cells or mathematical models. While individual neurons are simple, many of them together in a network can perform complex tasks. There are two main types of neural networks.In neuroscience, a biological neural network is a physical structure found in brains and complex nervous systems – a population of nerve cells connected by synapses.
In machine learning, an artificial neural network is a mathematical model used to approximate nonlinear functions. Artificial neural networks are used to solve artificial intelligence problems.
A quantum computer is a computer that exploits superposed and entangled states. Quantum computers can be viewed as sampling from quantum systems that evolve in ways that may be described as operating on an enormous number of possibilities simultaneously, though still subject to strict computational constraints. By contrast, ordinary ("classical") computers operate according to deterministic rules. It is widely believed that a quantum computer could perform some calculations exponentially faster than any classical computer. For example, a large-scale quantum computer could break some widely used public-key cryptographic schemes and aid physicists in performing physical simulations. However, current hardware implementations of quantum computation are largely experimental and only suitable for specialized tasks.
A blockchain is a distributed ledger with growing lists of records (blocks) that are securely linked together via cryptographic hashes. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. Since each block contains information about the previous block, they effectively form a chain, with each additional block linking to the ones before it. Consequently, blockchain transactions are resistant to alteration because, once recorded, the data in any given block cannot be changed retroactively without altering all subsequent blocks and obtaining network consensus to accept these changes.
Cryptography, or cryptology, is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties or the public from reading private messages. Modern cryptography exists at the intersection of the disciplines of mathematics, computer science, information security, electrical engineering, digital signal processing, physics, and others. Core concepts related to information security are also central to cryptography. Practical applications of cryptography include electronic commerce, chip-based payment cards, digital currencies, computer passwords and military communications.
Network, networking and networked may refer to:.
An operating system (OS) is system software that manages computer hardware and software resources, and provides common services for computer programs.
Cloud computing is defined by the ISO as "a paradigm for enabling network access to a scalable and elastic pool of shareable physical or virtual resources with self-service provisioning and administration on demand". It is commonly referred to as "the cloud".
In computing, a database is an organized collection of data or a type of data store based on the use of a database management system (DBMS), the software that interacts with end users, applications, and the database itself to capture and analyze the data. The DBMS additionally encompasses the core facilities provided to administer the database. The sum total of the database, the DBMS and the associated applications can be referred to as a database system. Often the term "database" is also used loosely to refer to any of the DBMS, the database system or an application associated with the database.
Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch in biology because heredity is vital to organisms' evolution. Gregor Mendel, a Moravian Augustinian friar working in the 19th century in Brno, was the first to study genetics scientifically. Mendel studied "trait inheritance", patterns in the way traits are handed down from parents to offspring over time. He observed that organisms inherit traits by way of discrete "units of inheritance". This term, still used today, is a somewhat ambiguous definition of what is referred to as a gene.
Neuroscience is the scientific study of the nervous system, its functions, and its disorders. It is a multidisciplinary science that combines physiology, anatomy, molecular biology, developmental biology, cytology, psychology, physics, computer science, chemistry, medicine, statistics, and mathematical modeling to understand the fundamental and emergent properties of neurons, glia, and neural circuits. The understanding of the biological basis of learning, memory, behavior, perception, and consciousness has been described by Eric Kandel as the "epic challenge" of the biological sciences.
Medicine is the science and practice of caring for patients, managing the diagnosis, prognosis, prevention, treatment and palliation of their injury or disease, while promoting their health. Medicine encompasses a variety of health care practices which evolved to maintain and restore health through the prevention and treatment of illness. Contemporary medicine applies biomedical sciences, biomedical research, genetics, and medical technology to diagnose, treat, and prevent injury and disease, typically through various pharmaceuticals or surgery, but also through therapies such as psychotherapy, external splints and traction, medical devices, biologics, and ionizing radiation, amongst others.
Public Health may refer to:Public health, promoting health through organized efforts and informed choices of society and individuals
Public Health (journal), published by Elsevier for the Royal Society for Public Health
Public Health a 2021 proposed comedy television series by Rob Tepper
Public Health, a May 22, 2014 episode of Debatten, a Norwegian television series
Public Health, a July 6, 2000 episode of Today's Environment, television series by Five Star Productions.
Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people.
Ethics is the philosophical study of moral phenomena. Also called moral philosophy, it investigates normative questions about what people ought to do or which behavior is morally right. Its main branches include normative ethics, applied ethics, and metaethics.
Business is the practice of making one's living or making money by producing or buying and selling products. It is also "any activity or enterprise entered into for profit.".
Finance refers to monetary resources and to the study and discipline of money, currency, assets and liabilities. As a subject of study, it is a field of business administration which involves the planning, organizing, leading, and controlling of an organization's resources to achieve its goals. Based on the scope of financial activities in financial systems, the discipline can be divided into personal, corporate, and public finance.
Marketing is the act of acquiring, satisfying and retaining customers. It is one of the primary components of business management and commerce.
Entrepreneurship is the creation or extraction of economic value by identifying and commercializing opportunities to deliver products or services, a process that typically requires considerable initiative and bears risk. This process may also encompass the pursuit of values that extend beyond mere economic considerations.
Geopolitics is the study of the effects of Earth's geography on politics and international relations. Geopolitics usually refers to countries and relations between them. According to multiple researchers, the term is currently being used to describe a broad spectrum of concepts, in a general sense used as "a synonym for international political relations", but more specifically "to imply the global structure of such relations"; this usage builds on an "early-twentieth-century term for a pseudoscience of political geography" and other pseudoscientific theories of historical and geographic determinism.
Climatology or climate science is the scientific study of Earth's climate, typically defined as weather conditions averaged over a period of at least 30 years. Climate concerns the atmospheric condition during an extended to indefinite period of time; weather is the condition of the atmosphere during a relative brief period of time. The main topics of research are the study of climate variability, mechanisms of climate changes and modern climate change. This topic of study is regarded as part of the atmospheric sciences and a subdivision of physical geography, which is one of the Earth sciences. Climatology includes some aspects of oceanography and biogeochemistry.
Failed to fetch Energy Systems.
Environmental science is an academic field that integrates the physical, biological, and mathematical sciences to study the environment and solve environmental problems. It uses an integrated, quantitative, and interdisciplinary approach to analyze environmental systems and emerged from the fields of natural history and medicine during the Enlightenment. It is considered interdisciplinary because it is an integration of various fields such as: biology, chemistry, physics, geology, engineering, sociology, and ecology.
Astronautics is the practice of sending spacecraft beyond Earth's atmosphere into outer space. Spaceflight is one of its main applications and space science is its overarching field.
Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics.
Automation describes a wide range of technologies that reduce human intervention in processes, mainly by predetermining decision criteria, subprocess relationships, and related actions, as well as embodying those predeterminations in machines. Automation has been achieved by various means including mechanical, hydraulic, pneumatic, electrical, electronic devices, and computers, usually in combination. Complicated systems, such as modern factories, airplanes, and ships typically use combinations of all of these techniques. The benefits of automation includes labor savings, reducing waste, savings in electricity costs, savings in material costs, and improvements to quality, accuracy, and precision.
Biotechnology is a multidisciplinary field that involves the integration of natural sciences and engineering sciences in order to achieve the application of organisms and parts thereof for products and services. Specialists in the field are known as biotechnologists.
Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter. This definition of nanotechnology includes all types of research and technologies that deal with these special properties. It is common to see the plural form "nanotechnologies" as well as "nanoscale technologies" to refer to research and applications whose common trait is scale. An earlier understanding of nanotechnology referred to the particular technological goal of precisely manipulating atoms and molecules for fabricating macroscale products, now referred to as molecular nanotechnology.
Materials science is an interdisciplinary field of researching and discovering materials. Materials engineering is an engineering field of finding uses for materials in other fields and industries.
Cognitive science is the interdisciplinary, scientific study of the mind and its processes. It examines the nature, the tasks, and the functions of cognition. Mental faculties of concern to cognitive scientists include perception, memory, attention, reasoning, language, and emotion. To understand these faculties, cognitive scientists borrow from fields such as psychology, philosophy, artificial intelligence, neuroscience, linguistics, and anthropology. The typical analysis of cognitive science spans many levels of organization, from learning and decision-making to logic and planning; from neural circuitry to modular brain organization. One of the fundamental concepts of cognitive science is that "thinking can best be understood in terms of representational structures in the mind and computational procedures that operate on those structures.".
Game theory is the study of mathematical models of strategic interactions. It has applications in many fields of social science, and is used extensively in economics, logic, systems science and computer science. Initially, game theory addressed two-person zero-sum games, in which a participant's gains or losses are exactly balanced by the losses and gains of the other participant. In the 1950s, it was extended to the study of non zero-sum games, and was eventually applied to a wide range of behavioral relations. It is now an umbrella term for the science of rational decision making in humans, animals, and computers.
Information theory is the mathematical study of the quantification, storage, and communication of a particular type of mathematically defined information. The field was established and formalized by Claude Shannon in the 1940s, though early contributions were made in the 1920s through the works of Harry Nyquist and Ralph Hartley. It is at the intersection of electronic engineering, mathematics, statistics, computer science, neurobiology, physics, and electrical engineering.
Employment is a relationship between two parties regulating the provision of paid labour services. Usually based on a contract, one party, the employer, which might be a corporation, a not-for-profit organization, a co-operative, or any other entity, pays the other, the employee, in return for carrying out assigned work. Employees work in return for wages, which can be paid on the basis of an hourly rate, by piecework or an annual salary, depending on the type of work an employee does, the prevailing conditions of the sector and the bargaining power between the parties. Employees in some sectors may receive gratuities, bonus payments or stock options. In some types of employment, employees may receive benefits in addition to payment. Benefits may include health insurance, housing, and disability insurance.
Education is the transmission of knowledge and skills and the development of character traits. Formal education happens in a complex institutional framework, like public schools. Non-formal education is also structured but takes place outside the formal schooling system, while informal education is unstructured learning through daily experiences. Formal and non-formal education are divided into levels that include early childhood education, primary education, secondary education, and tertiary education. Other classifications focus on the teaching method, like teacher-centered and student-centered education, and on the subject, like science education, language education, and physical education. The term "education" can also refer to the mental states and qualities of educated people and the academic field studying educational phenomena.
Sleep is a state of reduced mental and physical activity in which consciousness is altered and certain sensory activity is inhibited. During sleep, there is a marked decrease in muscle activity and interactions with the surrounding environment. While sleep differs from wakefulness in terms of the ability to react to stimuli, it still involves active brain patterns, making it more reactive than a coma or disorders of consciousness.
A hobby is considered to be a regular activity that is done for enjoyment, typically during one's leisure time. Hobbies include collecting themed items and objects, engaging in creative and artistic pursuits, playing sports, or pursuing other amusements or avocations. Participation in hobbies encourages acquiring substantial skills and knowledge in that area. A list of hobbies changes with renewed interests and developing fashions, making it diverse and lengthy. Hobbies tend to follow trends in society. For example, stamp collecting was popular during the nineteenth and twentieth centuries as postal systems were the main means of communication; as of 2024, video games became more popular following technological advances. The advancing production, technology, and labour movements of the nineteenth century provided workers with more leisure time to engage in hobbies. Because of this, the efforts of people investing in hobbies has increased with time.
Shopping is an activity in which a customer browses the available goods or services presented by one or more retailers with the potential intent to purchase a suitable selection of them. A typology of shopper types has been developed by scholars which identifies one group of shoppers as recreational shoppers, that is, those who enjoy shopping and view it as a leisure activity.
Health has a variety of definitions, which have been used for different purposes over time. In general, it refers to physical and emotional well-being, especially that associated with normal functioning of the human body, absent of disease, pain, or injury.
Family is a group of people related either by consanguinity or affinity. It forms the basis for social order. Ideally, families offer predictability, structure, and safety as members mature and learn to participate in the community. Historically, most human societies use family as the primary purpose of attachment, nurturance, and socialization.
Leisure has often been defined as a quality of experience or as free time. Free time is time spent away from business, work, job hunting, domestic chores, and education, as well as necessary activities such as eating and sleeping. Leisure as an experience usually emphasizes dimensions of perceived freedom and choice. It is done for "its own sake", for the quality of experience and involvement. Other classic definitions include Thorstein Veblen's (1899) of "nonproductive consumption of time." Free time is not easy to define due to the multiplicity of approaches used to determine its essence. Different disciplines have definitions reflecting their common issues: for example, sociology on social forces and contexts and psychology as mental and emotional states and conditions. From a research perspective, these approaches have an advantage of being quantifiable and comparable over time and place.
A grocery store (AE), grocery shop or grocer's shop (BE) or simply grocery is a retail store that primarily retails a general range of food products, which may be fresh or packaged. In everyday US usage, however, "grocery store" is a synonym for supermarket, and is not used to refer to other types of stores that sell groceries. In the UK, shops that sell food are distinguished as grocers or grocery shops.
Physical fitness is a state of health and well-being and, more specifically, the ability to perform aspects of sports, occupations, and daily activities. Physical fitness is generally achieved through proper nutrition, moderate-vigorous physical exercise, and sufficient rest along with a formal recovery plan.
Cleaning is the process of removing unwanted substances, such as dirt, dust, and other impurities, from an object or environment. Cleaning is often performed for aesthetic, hygienic, functional, safety, or environmental protection purposes. Cleaning occurs in many different contexts, and uses many different methods. Several occupations are devoted to cleaning.
Laundry is the washing of clothing and other textiles, and, more broadly, their drying and ironing as well. Laundry has been part of history since humans began to wear clothes, so the methods by which different cultures have dealt with this universal human need are of interest to several branches of scholarship.
Personal finance is the financial management that an individual or a family unit performs to budget, save, and spend monetary resources in a controlled manner, taking into account various financial risks and future life events.
Telecommunication, often used in its plural form or abbreviated as telecom, is the transmission of information over a distance using electrical or electronic means, typically through cables, radio waves, or other communication technologies. These means of transmission may be divided into communication channels for multiplexing, allowing for a single medium to transmit several concurrent communication sessions. Long-distance technologies invented during the 19th, 20th and 21st centuries generally use electric power, and include the electrical telegraph, telephone, television, and radio.
Social media are new media technologies that facilitate the creation, sharing and aggregation of content amongst virtual communities and networks. Common features include:Online platforms enable users to create and share content and participate in social networking.
User-generated content—such as text posts or comments, digital photos or videos, and data generated through online interactions.
Service-specific profiles that are designed and maintained by the social media organization.
Social media helps the development of online social networks by connecting a user's profile with those of other individuals or groups.
Television (TV) is a telecommunication medium for transmitting moving images and sound. Additionally, the term can refer to a physical television set rather than the medium of transmission. Television is a mass medium for advertising, entertainment, news, and sports. The medium is capable of more than "radio broadcasting", which refers to an audio signal sent to radio receivers.
A birthday is the anniversary of the birth of a person or the figurative birth of an institution. Birthdays of people are celebrated in numerous cultures, often with birthday gifts, birthday cards, a birthday party, or a rite of passage.
A wedding is a ceremony in which two people are united in marriage. Wedding traditions and customs vary greatly between cultures, ethnicities, races, religions, denominations, countries, social classes, and sexual orientations. Most wedding ceremonies involve an exchange of marriage vows by a couple; a presentation of a gift ; and a public proclamation of marriage by an authority figure or celebrant. Special wedding garments are often worn, and the ceremony is sometimes followed by a wedding reception. Music, poetry, prayers, or readings from religious texts or literature are also commonly incorporated into the ceremony, as well as superstitious customs.
A funeral is a ceremony connected with the final disposition of a corpse, such as a burial, entombment or cremation with the attendant observances. Funerary customs comprise the complex of beliefs and practices used by a culture to remember and respect the dead, from interment, to various monuments, prayers, and rituals undertaken in their honour. Customs vary between cultures and religious groups. Funerals have both normative and legal components. Common secular motivations for funerals include mourning the deceased, celebrating their life, and offering support and sympathy to the bereaved; additionally, funerals may have religious aspects that are intended to help the soul of the deceased reach the afterlife, resurrection or reincarnation.
Religion is a range of social-cultural systems, including designated behaviors and practices, ethics, morals, beliefs, worldviews, texts, sanctified places, prophecies, or organizations, that generally relate humanity to supernatural, transcendental, and spiritual elements—although there is no scholarly consensus over what precisely constitutes a religion. It is an essentially contested concept. Different religions may or may not contain various elements ranging from the divine, sacredness, faith, and a supernatural being or beings.
Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of status or resources.
The branch of social science that studies politics and government is referred to as political science.
History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past.
Geography is the study of the lands, features, inhabitants, and phenomena of Earth. Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science. Geography has been called "a bridge between natural science and social science disciplines.".
Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine.
Technology is the application of conceptual knowledge to achieve practical goals, especially in a reproducible way. The word technology can also mean the products resulting from such efforts, including both tangible tools such as utensils or machines, and intangible ones such as software. Technology plays a critical role in science, engineering, and everyday life.
Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty.
Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role.
Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions.
Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups.
Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method.
Economics is a social science that studies the production, distribution, and consumption of goods and services.
Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people.
An apple is the round, edible fruit of an apple tree. Fruit trees of the orchard or domestic apple, the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies and religions.
A banana is an elongated, edible fruit—botanically a berry—produced by several kinds of large treelike herbaceous flowering plants in the genus Musa. In some countries, cooking bananas are called plantains, distinguishing them from dessert bananas. The fruit is variable in size, color and firmness, but is usually elongated and curved, with soft flesh rich in starch covered with a peel, which may have a variety of colors when ripe. It grows upward in clusters near the top of the plant. Almost all modern edible seedless (parthenocarp) cultivated bananas come from two wild species – Musa acuminata and Musa balbisiana, or their hybrids.
Orange most often refers to:Orange (fruit), the fruit of the tree species  Citrus × sinensis
Orange blossom, its fragrant flower
Orange juice
Orange (colour), the color of an orange fruit, occurs between red and yellow in the visible light spectrum
Some other citrus or citrus-like fruit, see list of plants known as orange
Orange (word), both a noun and an adjective in the English language.
The garden strawberry is a widely grown hybrid plant cultivated worldwide for its fruit. The genus Fragaria, the strawberries, is in the rose family, Rosaceae. The fruit is appreciated for its aroma, bright red colour, juicy texture, and sweetness. It is eaten either fresh or in prepared foods such as jam, ice cream, and chocolates. Artificial strawberry flavourings and aromas are widely used in commercial products. Botanically, the strawberry is not a berry, but an aggregate accessory fruit. Each apparent 'seed' on the outside of the strawberry is actually an achene, a botanical fruit with a seed inside it.
Blueberries are a widely distributed and widespread group of perennial flowering plants with blue or purple berries. They are classified in the section Cyanococcus within the genus Vaccinium. Commercial blueberries—both wild (lowbush) and cultivated (highbush)—are all native to North America. The highbush varieties were introduced into Europe during the 1930s.
The raspberry is the edible fruit of several plant species in the genus Rubus of the rose family, most of which are in the subgenus Idaeobatus. The name also applies to these plants themselves. Raspberries are perennial with woody stems.
The blackberry is an edible fruit produced by many species in the genus Rubus in the family Rosaceae, hybrids among these species within the subgenus Rubus, and hybrids between the subgenera Rubus and Idaeobatus. The taxonomy of blackberries has historically been confused because of hybridization and apomixis so that species have often been grouped together and called species aggregates.
The pineapple is a tropical plant with an edible fruit; it is the most economically significant plant in the family Bromeliaceae.
A mango is an edible stone fruit produced by the tropical tree Mangifera indica. It originated in the northeastern part of the Indian subcontinent, in what is now Bangladesh, northeastern India and Myanmar. M. indica has been cultivated in South and Southeast Asia since ancient times, resulting in two modern mango cultivar lineages: the "Indian" and the "Southeast Asian" types. Other species in the genus Mangifera also produce edible fruits called "mangoes," most of which are found in the Malesian ecoregion.
The papaya, papaw, or pawpaw is the plant species Carica papaya, one of the 21 accepted species in the genus Carica of the family Caricaceae. Papaya is also the name of its fruit. It was first domesticated in Mesoamerica, within modern-day southern Mexico and Central America. It is grown in several countries in regions with a tropical climate. In 2024, India was the leading producer, accounting for 36% of the world total.
A grape is a fruit, botanically a berry, of the deciduous woody vines of the flowering plant genus Vitis. Grapes are a non-climacteric type of fruit, generally occurring in clusters.
The watermelon is a species of flowering plant in the family Cucurbitaceae, that has a large, edible fruit. It is a scrambling and trailing vine-like plant, and is widely cultivated worldwide, with more than 1,000 varieties.
The cantaloupe is a type of true melon with sweet, aromatic, and usually orange flesh. Originally, cantaloup referred to the true cantaloupe or European cantaloupe with non- to slightly netted and often ribbed rind. Today, it also refers to the muskmelon with strongly netted rind, which is called cantaloupe in North America, rockmelon in Australia and New Zealand, and spanspek in Southern Africa. Cantaloupes range in mass from 0.5 to 5 kilograms.
Honeydew may refer to:Honeydew (melon), a cultivar group of melon
Honeydew (secretion), a sugar-rich sticky substance secreted by various animals
Honeydew moth, a moth of Southern and Middle America
Honeydew, California, United States, a town
Honeydew, West Virginia, United States, an unincorporated community
Honeydew (color), a pale shade of the color spring green
Bunsen Honeydew, a fictional character from The Muppets franchise
Honeydew (album), a 2008 album by Shawn Mullins
Honeydew (film), a 2020 American horror film written and directed by Devereux Milburn
Honey Dew Donuts, a Massachusetts-based franchise selling donuts and other breakfast foods
Fuller's Organic Honey Dew, a brand of pale ale brewed by Fuller's Brewery
Simon "Honeydew" Lane, a member of internet gaming group The Yogscast
"Honeydew" , a 2023 episode of The Bear TV series

.
Kiwi most commonly refers to:Kiwi (bird), a flightless bird native to New Zealand
Kiwi (nickname), an informal name for New Zealanders
Kiwifruit, an edible hairy fruit with many seeds
Kiwi dollar or New Zealand dollar, a unit of currency.
The peach is a deciduous tree that bears edible juicy fruits with various characteristics. Most are simply called peaches, while the glossy-skinned, non-fuzzy varieties are called nectarines. Though from the same species, they are regarded commercially as different fruits.
A plum is a fruit of some species in Prunus subg. Prunus. Dried plums are usually called prunes.
A cherry is the fruit of many plants of the genus Prunus, and is a fleshy drupe.
An apricot is a fruit, or the tree that bears the fruit, of several species in the genus Prunus. Usually an apricot is from the species Prunus armeniaca, but the fruits of the other species in Prunus sect. Armeniaca are also called apricots. In 2023, world production of apricots was 3.7 million tonnes, led by Turkey with 20% of the total.
The pomegranate is a fruit-bearing, deciduous shrub in the family Lythraceae, subfamily Punicoideae, that grows to between 1.5–5 metres (5–16 ft) tall. Rich in symbolic and mythological associations in many cultures, it originated from the Iranian plateau including Iran, the Caucasus, Turkmenistan, Afghanistan and Pakistan. Pomegranate was first domesticated by ancient Iranians in the Persian plateau and nearby regions about 5,000 years ago. It is extensively cultivated for its fruit.
The tomato is a plant whose fruit is an edible berry that is eaten as a vegetable. The tomato is a member of the nightshade family that includes tobacco, potato, and chili peppers. It originated from western South America, and may have been domesticated there, in Mexico, or in Central America. The Spanish introduced tomatoes to Eurasia in the Columbian exchange in the 16th century.
The cucumber is a widely-cultivated creeping vine plant in the family Cucurbitaceae that bears cylindrical to spherical fruits, which are used as culinary vegetables. Considered an annual plant, there are three main types of cucumber—slicing, pickling, and seedless—within which several cultivars have been created. The cucumber originates in Asia extending from India, Nepal, Bangladesh, China, and Northern Thailand, but now grows on most continents, and many different types of cucumber are grown commercially and traded on the global market. In North America, the term wild cucumber refers to plants in the genera Echinocystis and Marah, though the two are not closely related.
The carrot is a root vegetable, typically orange in colour, though heirloom variants including purple, black, red, white, and yellow cultivars exist, all of which are domesticated forms of the wild carrot, Daucus carota, native to Europe and Southwestern Asia. The plant probably originated in Iran and was originally cultivated for its leaves and seeds.
Broccoli is an edible green plant in the cabbage family whose large flowering head, stalk and small associated leaves are eaten as a vegetable. Broccoli is classified in the Italica cultivar group of the species Brassica oleracea. Broccoli has large flower heads, or florets, usually dark green, arranged in a tree-like structure branching out from a thick stalk, which is usually light green. Leaves surround the mass of flower heads. Broccoli resembles cauliflower, a different but closely related cultivar group of the same Brassica species.
Cauliflower is one of several vegetables cultivated from the species Brassica oleracea in the genus Brassica, which is in the Brassicaceae family. Cauliflower usually grows with one main stem that carries a large, rounded "head" made of tightly clustered, immature white or off-white flower buds called the "curd". Typically, only the "head" is eaten.
Spinach is a leafy green flowering plant native to Central and Western Asia. It is of the order Caryophyllales, family Amaranthaceae, subfamily Chenopodioideae. Its leaves are a common vegetable consumed either fresh, cooked or after storage. The taste differs considerably between cooked and raw: the high oxalate content may be reduced by steaming.
Kale, also called leaf cabbage, belongs to a group of cabbage cultivars primarily grown for their edible leaves, but it is also used as an ornamental plant. Its multiple different cultivars vary quite a bit in appearance; the leaves can be bumpy, curly, or flat, and the color ranges from purple to green.
Lettuce is an annual plant of the family Asteraceae mostly grown as a leaf vegetable. The leaves are most often used raw in green salads, although lettuce is also seen in other kinds of food, such as sandwiches, wraps and soups; it can also be grilled. Its stem and seeds are sometimes used; celtuce is one variety grown for its stems, which are eaten either raw or cooked. In addition to its main use as a leafy green, it has also gathered religious and medicinal significance over centuries of human consumption. Europe and North America originally dominated the market for lettuce, but by the late 20th century the consumption of lettuce had spread throughout the world. In 2023, world production of lettuce was 28 million tonnes, led by China with 53% of the total.
Eruca sativa is an edible annual plant in the family Brassicaceae. Other common names include salad rocket, garden rocket, colewort, roquette, ruchetta, rucola, rucoli, and rugula.
Zucchini, courgette, or Cucurbita pepo var. cylindrica is a summer squash, a vining herbaceous plant whose fruit are harvested when their immature seeds and epicarp (rind) are still soft and edible. It is closely related, but not identical, to the marrow; its fruit may be called marrow when mature.
Eggplant, aubergine, brinjal, or baigan is a plant species in the nightshade family Solanaceae. Solanum melongena is grown worldwide for its edible fruit, typically used as a vegetable in cooking.
The bell pepper is the fruit of plants in the Grossum Group of the species Capsicum annuum. Cultivars of the plant produce fruits in different colors, including red, yellow, orange, green, white, and purple. Bell peppers are sometimes grouped with less pungent chili varieties as "sweet peppers". While they are botanically fruits—classified as berries—they are commonly used as a vegetable ingredient or side dish. Other varieties of the genus Capsicum are categorized as chili peppers when they are cultivated for their pungency, including some varieties of Capsicum annuum.
The onion, also known as the bulb onion or common onion, is a vegetable that is the most widely cultivated species of the genus Allium. The shallot is a botanical variety of the onion which was classified as a separate species until 2011. The onion's close relatives include garlic, scallion, leek, and chives.
Garlic is a species of bulbous flowering plants in the genus Allium. Its close relatives include the onion, shallot, leek, chives, Welsh onion, and Chinese onion. Garlic is native to central and western Asia, stretching from the Black Sea through the southern Caucasus, northeastern Iran, and the Hindu Kush. It has naturalized in many other parts of the world, including Mediterranean Europe and China. There are two subspecies and hundreds of varieties of garlic.
Ginger is a flowering plant whose rhizome, ginger root or ginger, is widely used as a spice and a folk medicine. It is an herbaceous perennial that grows annual pseudostems about one meter tall, bearing narrow leaf blades. The inflorescences bear flowers having pale yellow petals with purple edges, and arise directly from the rhizome on separate shoots.
The potato is a starchy tuberous vegetable native to the Americas that is consumed as a staple food in many parts of the world. Potatoes are underground stem tubers of the plant Solanum tuberosum, a perennial in the nightshade family Solanaceae.
The sweet potato or sweetpotato is a dicotyledonous plant in the morning glory family, Convolvulaceae. Its sizeable, starchy, sweet-tasting tuberous roots are used as a root vegetable, which is a staple food in parts of the world. Cultivars of the sweet potato have been bred to bear tubers with flesh and skin of various colors. Moreover, the young shoots and leaves are occasionally eaten as greens. The sweet potato and the potato are only distantly related, both being in the order Solanales. Although darker sweet potatoes are often known as yams in parts of North America, they are even more distant from actual yams, which are monocots in the order Dioscoreales.
Maize, also known as corn in North American English, is a tall stout grass that produces cereal grain. The leafy stalk of the plant gives rise to male inflorescences or tassels which produce pollen, and female inflorescences called ears. The ears yield grain, known as kernels or seeds. In modern commercial varieties, these are usually yellow or white; other varieties can be of many colors. Maize was domesticated by indigenous peoples in southern Mexico about 9,000 years ago from wild teosinte. Native Americans planted it alongside beans and squashes in the Three Sisters polyculture.
Could not find summary for "Green Beans".
Asparagus or garden asparagus is a perennial flowering plant species in the genus Asparagus native to Eurasia. Widely cultivated as a vegetable crop, its young shoots are used as a spring vegetable.
Celery is a cultivated plant belonging to the species Apium graveolens in the family Apiaceae that has been used as a vegetable since ancient times.
A mushroom is the fleshy, spore-bearing fruiting body of a fungus, typically produced above ground on soil or another food source. A toadstool generally refers to a poisonous mushroom.
The avocado, alligator pear or avocado pear is an evergreen tree in the laurel family (Lauraceae). It is native to the Americas, with archaeological evidence of early human avocado use dating back thousands of years across various regions of Central and South America. It was prized for its large and unusually oily fruit. The native range of avocado extends from Mexico to Peru, encompassing much of Central America and parts of northern and western South America.
Lime most commonly refers to:Lime (fruit), a green citrus fruit
Lime (material), inorganic materials containing calcium, usually calcium oxide or calcium hydroxide
Lime (color), a color between yellow and green.
The lemon is a species of small evergreen tree in the Citrus genus of the flowering plant family Rutaceae. A true lemon is a hybrid of the citron and the bitter orange. Its origins are uncertain, but some evidence suggests lemons originated during the 1st millennium BC in what is now northeastern India. Some other citrus fruits are called lemon.
The grapefruit is a subtropical citrus tree known for its relatively large, sour to semi-sweet, somewhat bitter fruit. The flesh of the fruit is segmented and varies in color from pale yellow to dark red.
Pears are fruits produced and consumed around the world, growing on a tree and are harvested in late summer into mid-autumn. The pear tree and shrub are a species of genus Pyrus, in the family Rosaceae, bearing the pomaceous fruit of the same name. Several species of pears are valued for their edible fruit and juices, while others are cultivated as trees.
The coconut is a member of the palm family (Arecaceae) and the only living species of the genus Cocos. The term "coconut" can denote the whole coconut palm tree or the large hard fruit. Originally native to Central Indo-Pacific, they are ubiquitous in coastal tropical regions.
Passiflora edulis, commonly known as passion fruit, is a vine species of passion flower. The fruit is a pepo, a type of botanical berry, round to oval, either yellow or dark purple at maturity, with a soft to firm, juicy interior filled with numerous seeds.
Lychee is a monotypic taxon and the sole member in the genus Litchi in the soapberry family, Sapindaceae.
The fruit is edible and has a sweet, mildly tart flavor and a distinctive floral aroma often described as rose-like.
The durian is the edible fruit of several tree species belonging to the genus Durio. There are 30 recognised species, at least nine of which produce edible fruit. Durio zibethinus, native to Borneo, Sumatra, and the Malay Peninsula, is the only species available on the international market. It has over 300 named varieties in Thailand and over 200 in Malaysia as of 2021. Other species are sold in their local regions.
Guava, also known as the 'guava-pear' in various regions, is a common tropical fruit cultivated in many tropical and subtropical regions. The common guava Psidium guajava is a small tree in the myrtle family (Myrtaceae), native to Mexico, Central America, the Caribbean and northern South America.
Carambola, also known as star fruit, is the fruit of Averrhoa carambola, a species of tree native to tropical Southeast Asia. The edible fruit has distinctive ridges running down its sides. When cut in cross-section, it resembles a star, giving it the name of star fruit. The entire fruit is edible, usually raw, and may be cooked or made into relishes, preserves, garnish, and juices. It is commonly consumed in Southeast Asia, South Asia, the South Pacific, Micronesia, parts of East Asia, the United States, parts of Latin America, and the Caribbean. The tree is cultivated throughout tropical areas of the world.
Pitaya, pitahaya or commonly known as dragon fruit is the fruit of several cactus species indigenous to the region of southern Mexico and along the Pacific coasts of Guatemala, Costa Rica, and El Salvador. Pitaya is cultivated in East Asia, South Asia, Southeast Asia, continental America, the Caribbean, Australia, Brazil, Madeira (Portugal), and throughout tropical and subtropical regions of the world.
Rice is a cereal grain and in its domesticated form is the staple food of over half of the world's population, particularly in Asia and Africa. Rice is the seed of the grass species Oryza sativa —or, much less commonly, Oryza glaberrima. Asian rice was domesticated in China some 13,500 to 8,200 years ago; African rice was domesticated in Africa about 3,000 years ago. Rice has become commonplace in many cultures worldwide; in 2023, 800 million tons were produced, placing it third after sugarcane and maize. Only some 8% of rice is traded internationally. China, India, and Indonesia are the largest consumers of rice. A substantial amount of the rice produced in developing nations is lost after harvest through factors such as poor transport and storage. Rice yields can be reduced by pests including insects, rodents, and birds, as well as by weeds, and by diseases such as rice blast. Traditional rice polycultures such as rice-duck farming, and modern integrated pest management seek to control damage from pests in a sustainable way.
Pasta is a type of food typically made from an unleavened dough of wheat flour mixed with water or eggs, and formed into sheets or other shapes, then cooked by boiling or baking. Pasta was originally only made with durum, although the definition has been expanded to include alternatives for a gluten-free diet, such as rice flour, or legumes such as beans or lentils. Pasta is believed to have developed independently in Italy and is a staple food of Italian cuisine, with evidence of Etruscans making pasta as early as 400 BCE in Italy.
Bread is a baked food product made from water, flour, and often yeast. It is a staple food across the world, particularly in Europe and the Middle East. Throughout recorded history and around the world, it has been an important part of many cultures' diets. It is one of the oldest human-made foods, having been of significance since the dawn of agriculture, and plays an essential role in both religious rituals and secular culture.
A tortilla is a thin, circular unleavened flatbread from Mesoamerica originally made from masa, and now also from wheat flour.
The oat, sometimes called the common oat, is a species of cereal grass (Avena) grown for fodder and for its seed, which is known by the same name. Oats appear to have been domesticated as a secondary crop, as their seeds resembled those of other cereals closely enough for them to be included by early cultivators. Oats tolerate cold winters less well than cereals such as wheat, barley, and rye, but need less summer heat and more rain, making them important in areas such as Northwest Europe that have cool, wet summers. They can tolerate low-nutrient and acid soils. Oats grow thickly and vigorously, allowing them to outcompete many weeds, and compared to other cereals are relatively free from diseases.
Quinoa is a flowering plant in the amaranth family. It is a herbaceous annual plant grown as a crop primarily for its edible seeds; the seeds are high in protein, dietary fiber, B vitamins and dietary minerals especially potassium and magnesium in amounts greater than in many grains. Quinoa is not a grass but rather a pseudocereal botanically related to spinach and amaranth, and originated in the Andean region of northwestern South America. It was first used to feed livestock 5,200–7,000 years ago, and for human consumption 3,000–4,000 years ago in the Lake Titicaca basin of Bolivia and Peru.
Barley, a member of the grass family, is a major cereal grain grown in temperate climates globally. One of the first cultivated grains, it was domesticated in the Fertile Crescent around 9000 BC, giving it nonshattering spikelets and making it much easier to harvest. Its use then spread throughout Eurasia by 2000 BC. Barley prefers relatively low temperatures and well-drained soil to grow. It is relatively tolerant of drought and soil salinity, but is less winter-hardy than wheat or rye.
The lentil is an annual legume grown for its lens-shaped edible seeds or pulses, also called lentils. It is about 40 cm (16 in) tall, and the seeds grow in pods, usually with two seeds in each.
The chickpea or chick pea is an annual legume of the family Fabaceae, subfamily Faboideae, cultivated for its edible seeds. Its different types are variously known as gram, Bengal gram, chana dal, garbanzo, garbanzo bean, or Egyptian pea. It is one of the earliest cultivated legumes, the oldest archaeological evidence of which was found in Syria.
Could not find summary for "Black Beans".
The kidney bean is a variety of the common bean ; it has such a common name owing to its resemblance to a human kidney.
Tofu  or bean curd is a food prepared by pressing the curds of coagulated soy milk into solid white blocks of varying softness: silken, soft, firm, and extra firm.
Tempeh or tempe is a traditional Indonesian food made from fermented soybeans. It is made by a natural culturing and controlled fermentation process that binds soybeans into a cake form. A fungus, Rhizopus oligosporus or Rhizopus oryzae, is used in the fermentation process and is also known as tempeh starter.
The chicken is a domesticated form of the red junglefowl, originally native to Southeast Asia. It was first domesticated around 8,000 years ago and is one of the most common and widespread domesticated animals in the world. Chickens are primarily kept for their meat and eggs, though they are also kept as pets.
Beef is the culinary name for meat from cattle. Beef can be prepared in various ways; cuts are often used for steak, which can be cooked to varying degrees of doneness, while trimmings are often ground or minced, as found in most hamburgers. Beef contains protein, iron, and vitamin B12. Along with other kinds of red meat, high consumption is associated with an increased risk of colorectal cancer and cardiovascular disease, especially when processed. Beef has a high environmental impact, being a primary driver of deforestation with the highest greenhouse gas emissions of any agricultural product.
Pork is the culinary name for the meat of the pig. It is the second most commonly consumed type of meat worldwide, following poultry, with evidence of pig husbandry dating back to 8000–9000 BCE.
Turkey, officially the Republic of Türkiye, is a country mainly located in Anatolia in West Asia, with a smaller part called East Thrace in Southeast Europe. It borders the Black Sea to the north; Georgia, Armenia, Azerbaijan, and Iran to the east; Iraq, Syria, and the Mediterranean Sea to the south; and the Aegean Sea, Greece, and Bulgaria to the west. Turkey is home to over 86 million people; most are ethnic Turks, while Kurds are the largest ethnic minority. Officially a secular state, Turkey has a Muslim-majority population. Ankara is Turkey's capital and second-largest city. Istanbul is its largest city and economic center. Other major cities include İzmir, Bursa, and Antalya.
Salmon are any of several commercially important species of euryhaline ray-finned fish from the genera Salmo and Oncorhynchus of the family Salmonidae, native to tributaries of the North Atlantic (Salmo) and North Pacific (Oncorhynchus) basins. Salmon is a colloquial or common name used for fish in this group, but is not a scientific name. Other closely related fish in the same family include trout, char, grayling, whitefish, lenok and taimen, all coldwater fish of the subarctic and cooler temperate regions with some sporadic endorheic populations in Central Asia.
A tuna is a saltwater fish that belongs to the tribe Thunnini, a subgrouping of the Scombridae (mackerel) family. The Thunnini comprise 15 species across five genera, the sizes of which vary greatly, ranging from the bullet tuna up to the Atlantic bluefin tuna, which averages 2 m (6.6 ft) and is believed to live up to 50 years.
A shrimp is a common name typically used for crustaceans with an elongated body and a primarily swimming mode of locomotion – usually decapods belonging to the Caridea or Dendrobranchiata, although some crustaceans outside of this order are also referred to as "shrimp".
Crabs are decapod crustaceans, either the Brachyura or various groups within the closely related Anomura, characterised by having a heavily armoured shell, their tail segments concealed under the body, the ability to run sideways, and the habit of hiding in rocky crevices. They do not form a single natural group or clade, but have convergently evolved multiple times from the ancestral decapod body plan through carcinisation, the process of creating this set of characteristics. As a group, they are thus polyphyletic, meaning they have multiple evolutionary origins.
Lobsters are malacostracan decapod crustaceans of the family Nephropidae or its synonym Homaridae. They have long bodies with muscular tails and live in crevices or burrows on the sea floor. Three of their five pairs of legs have claws, including the first pair, which are usually much larger than the others. Highly prized as seafood, lobsters are economically important and are often one of the most profitable commodities in the coastal areas they populate.
An egg is an organic vessel in which an embryo begins to develop.
Milk is a usually white liquid food produced by the mammary glands of lactating mammals. It is the primary source of nutrition for young mammals before they are able to digest solid food. Milk contains many nutrients, including calcium and protein, as well as lactose and saturated fat; the enzyme lactase is needed to break down lactose. Immune factors and immune-modulating components in milk contribute to milk immunity. The first milk, which is called colostrum, contains antibodies and immune-modulating components that strengthen the immune system against many diseases.
Cheddar cheese is a natural cheese that is relatively hard, off-white, and sometimes sharp-tasting. It originates from the village of Cheddar in Somerset, South West England.
Mozzarella is a semi-soft non-aged cheese prepared using the pasta filata ('stretched-curd') method. It originated in southern Italy.
Yogurt is a food produced by bacterial fermentation of milk. Fermentation of sugars in the milk by these bacteria produces lactic acid, which acts on milk protein to give yogurt its texture and characteristic tart flavor. Cow's milk is most commonly used to make yogurt. Milk from water buffalo, goats, ewes, mares, camels, and yaks is also used to produce yogurt. The milk used may be homogenized or not. It may be pasteurized or raw. Each type of milk produces substantially different results.
Butter is a dairy product made from the fat and protein components of churned cream. It is a semi-solid emulsion at room temperature, consisting of approximately 81% butterfat. It is used at room temperature as a spread, melted as a condiment, and used as a fat in baking, sauce-making, pan frying, and other cooking procedures.
The almond is a species of tree from the genus Prunus. Along with the peach, it is classified in the subgenus Amygdalus, distinguished from the other subgenera by corrugations on the shell (endocarp) surrounding the seed.
A walnut is the edible seed of any tree of the genus Juglans, particularly the Persian or English walnut, Juglans regia. They are accessory fruit because the outer covering of the fruit is technically an involucre and thus not morphologically part of the carpel; this means it cannot be a drupe but is instead a drupe-like nut.
Cashew is the common name of a tropical evergreen tree Anacardium occidentale, in the family Anacardiaceae. It is the source of the cashew nut and the cashew apple. The tree can grow as tall as 14 meters.
Peanuts is a syndicated daily and Sunday American comic strip written and illustrated by Charles M. Schulz. The strip originally ran from 1950 to 2000, continuing in reruns afterward. Peanuts is regarded as one of the most popular and influential comic strips in history, with 17,897 strips published in all, making it "arguably the longest story ever told by one human being". At the time of Schulz's death in 2000, Peanuts ran in over 2,600 newspapers, with a readership of roughly 355 million across 75 countries, and had been translated into 21 languages. It helped to cement the four-panel gag strip as the standard in the United States, and together with its merchandise earned Schulz more than $1 billion. Following successful animated television and stage-theatrical adaptations over the years, five animated theatrical films have been released.
Sunflower seeds are the seeds of the sunflower (Helianthus).
Could not find summary for "Pumpkin Seeds".
Olive oil is a vegetable oil obtained by pressing whole olives and extracting the oil.
Honey is a sweet and viscous substance made by several species of bees, the best-known of which are honey bees. Honey is made and stored to nourish bee colonies. Bees produce honey by gathering and then refining the sugary secretions of plants or the secretions of other insects, like the honeydew of aphids. This refinement takes place both within individual bees, through regurgitation and enzymatic activity, and during storage in the hive, through water evaporation that concentrates the honey's sugars until it is thick and viscous.
Maple syrup is a sweet syrup made from the sap of maple trees. In cold climates these trees store starch in their trunks and roots before winter; the starch is then converted to sugar that rises in the sap in late winter and early spring. Maple trees are tapped by drilling holes into their trunks and collecting the sap, which is heated to evaporate much of the water, leaving the concentrated syrup.
Chocolate is a food made from roasted and ground cocoa beans that can be a liquid, solid, or paste, either by itself or to flavor other foods. Cocoa beans are the processed seeds of the cacao tree. They are usually fermented to develop the flavor, then dried, cleaned, and roasted. The shell is removed to reveal nibs, which are ground to chocolate liquor The liquor can be processed to separate its two components, cocoa solids and cocoa butter, or shaped and sold as unsweetened baking chocolate. By adding sugar, sweetened chocolates are produced, which can be sold simply as dark chocolate, or, with the addition of milk, can be made into milk chocolate. Making milk chocolate with cocoa butter and without cocoa solids produces white chocolate.
Vanilla is a spice derived from orchids of the genus Vanilla, primarily obtained from the seed pods of the flat-leaved New World vanilla (V. planifolia).
Cinnamon is a spice obtained from the inner bark of several tree species from the genus Cinnamomum. Cinnamon is used mainly as an aromatic condiment and flavouring additive in a wide variety of cuisines, in particular sweet and savoury dishes such as biscuits, breakfast cereals, snack foods, bagels, teas, hot chocolate, and traditional foods. The aroma and flavour of cinnamon derive from its essential oil and principal component, cinnamaldehyde, as well as numerous other constituents, including eugenol.
Basil, also called great basil, is a culinary herb of the family Lamiaceae (mints). It is a tender plant, and is used in cuisines worldwide. In Western cuisine, the generic term "basil" refers to the variety also known as Genovese basil or sweet basil. Basil is native to tropical regions from Central Africa to Southeast Asia. In temperate climates basil is treated as an annual plant, but it can be grown as a short-lived perennial or biennial in warmer horticultural zones with tropical or Mediterranean climates.
Oregano is a species of flowering plant in the mint family, Lamiaceae. It was native to the Mediterranean region, but widely naturalised elsewhere in the temperate Northern Hemisphere.
Parsley, or garden parsley, is a species of flowering plant in the family Apiaceae that is native to the Balkans. It has been introduced and naturalized in Europe and elsewhere in the world with suitable climates, and is widely cultivated as a herb and a vegetable.
Mint or The Mint may refer to:.
Salvia rosmarinus, synonym Rosmarinus officinalis, commonly known as rosemary, is a shrub with fragrant, evergreen, needle-like leaves and purple or sometimes white, pink, or blue flowers. It is a member of the mint family, Lamiaceae.
Thyme is a culinary herb consisting of the dried aerial parts of some members of the genus Thymus of flowering plants in the mint family Lamiaceae. Thymes are native to Eurasia and north Africa. Thymes have culinary, medicinal, and ornamental uses. The species most commonly cultivated and used for culinary purposes is Thymus vulgaris, native to Southeast Europe.
A telephone, commonly shortened to phone, is a telecommunications device that enables two or more users to conduct a conversation when they are too far apart to be easily heard directly. A telephone converts sound, typically and most efficiently the human voice, into electronic signals that are transmitted via cables and other communication channels to another telephone which reproduces the sound to the receiving user. The term is derived from Ancient Greek: τῆλε, romanized: tēle, lit. 'far' and φωνή, together meaning distant voice.
A laptop is a portable personal computer (PC). Laptops typically have a clamshell form factor with a flat-panel screen on the inside of the upper lid and an alphanumeric keyboard and pointing device on the inside of the lower lid. Most of the computer's internal hardware is in the lower part, under the keyboard, although many modern laptops have a built-in webcam at the top of the screen, and some even feature a touchscreen display. In most cases, unlike tablet computers which run on mobile operating systems, laptops tend to run on desktop operating systems, which were originally developed for desktop computers.
Tablet may refer to:.
Keyboard may refer to:.
A mouse is a small rodent. Characteristically, mice are known to have a pointed snout, small rounded ears, a body-length scaly tail, and a high breeding rate. The best known mouse species is the common house mouse. Mice are also popular as pets. In some places, certain kinds of field mice are locally common. They are known to invade homes for food and shelter.
Monitor or monitor may refer to:.
Headphones are a pair of small loudspeaker drivers worn on or around the head over a user's ears. They are electroacoustic transducers, which convert an electrical signal to a corresponding sound. Headphones let a single user listen to an audio source privately, in contrast to a loudspeaker, which emits sound into the open air for anyone nearby to hear. Headphones are also known as earphones or, colloquially, cans. Circumaural and supra-aural headphones use a band over the top of the head to hold the drivers in place. Another type, known as earbuds or earpieces, consists of individual units that plug into the user's ear canal; within that category have been developed cordless air buds using wireless technology. A third type are bone conduction headphones, which typically wrap around the back of the head and rest in front of the ear canal, leaving the ear canal open. In the context of telecommunication, a headset is a combination of a headphone and microphone.
Charger or Chargers may refer to:.
A backpack, also called knapsack, schoolbag, rucksack, pack, booksack, bookbag, haversack, packsack, or backsack, is in its simplest frameless form, a fabric sack carried on one’s back and secured with two straps that go over the shoulders, and is used to carry goods from one place to another. It can feature an external or internal frame to transfer heavy loads off the user’s shoulders and onto their hips, reducing strain and increasing comfort on long hikes with heavy gear.
A wallet is a flat case or pouch, often used to carry small personal items such as physical currency, debit cards, and credit cards; identification documents such as driving licence, identification card, club card; photographs, transit pass, business cards and other paper or laminated cards. Wallets are generally made of fabric or leather, and they are usually pocket-sized and foldable.
Key, Keys, The Key or The Keys may refer to:.
A pen is a common writing instrument that applies ink to a surface, typically paper, for writing or drawing. Early pens such as reed pens, quill pens, dip pens and ruling pens held a small amount of ink on a nib or in a small void or cavity that had to be periodically recharged by dipping the tip of the pen into an inkwell.
Today, such pens find only a small number of specialized uses, such as in illustration and calligraphy. Reed pens, quill pens and dip pens, which were used for writing, have been replaced by ballpoint pens, rollerball pens, fountain pens and felt or ceramic tip pens.
A pencil is a writing or drawing implement with a solid pigment core in a protective casing that reduces the risk of core breakage and keeps it from marking the user's hand.
A notebook is a book or stack of paper pages that are often ruled and used for purposes such as note-taking, journaling, or other writing, drawing, or scrapbooking and more.



Paper is a thin sheet of matted cellulose fibers. Largely derived from lignocellulose, paper is created from a pulp dissolved into a slurry that is drained and dried into sheets. Different types of paper are defined by constituent fiber, paper pulp, sizing, coating, paper size, paper density and grammage.
An eraser is an article of stationery that is used for removing marks from paper or skin. Erasers have a rubbery consistency and come in a variety of shapes, sizes, and colors. Some pencils have an eraser on one end. Less expensive erasers are made from synthetic rubber and synthetic soy-based gum, but more expensive or specialized erasers are made from vinyl, plastic, or gum-like materials.
A highlighter, also called a fluorescent pen, is a type of writing device used to bring attention to sections of text by marking them with a vivid, translucent colour.
A typical highlighter is fluorescent yellow, with the colour coming from pyranine. Different compounds, such as rhodamines are used for other colours.
A ruler is an instrument used to make length measurements, whereby a length is read from a series of markings called "rules" along an edge of the device. Alternatively, it is called a rule, scale, line gauge, or metre/meter stick. Usually, the instrument is rigid and the edge itself is a straightedge, which additionally allows one to draw straighter lines. Rulers are an important tool in geometry, geography and mathematics. They have been used since at least 2650 BC.
Scissors or shears are hand-operated cutting tools that consists of a pair of pivoting blades whose sharpened edges slide firmly against and past each other when the handles (shank) on the opposite side of the pivot are squeezed shut, causing the target material in between the blades to be divided by the combined effort of both cutting and shearing. Scissors are usually used for cutting thin materials such as paper, cardboard, metal foil, cloth, rope and wire, although a large variety of scissors/shears exist for specialized purposes, and their design details often dictate which is best for the intended job.
Tape or Tapes may refer to:.
A stapler is a mechanical device that joins pages of paper or similar material together by driving a thin metal staple through the sheets and folding the ends. Staplers are widely used in government, business, offices, workplaces, homes, and schools.
A mug is a type of cup, a drinking vessel usually intended for hot drinks such as coffee, hot chocolate, or tea. Mugs have handles and usually hold a larger amount of fluid than other types of cups such as teacups or coffee cups. Typically, a mug holds approximately 250–350 ml (8–12 US fl oz) of liquid. A mug-shaped vessel much larger than this tends to be called a tankard.
A cup is a small container used to hold liquids for drinking, typically with a flattened hemispherical shape and an open "mouth", and often with a capacity of about 6–16 US fluid ounces (177–473 ml). Cups may be made of pottery, glass, metal, wood, stone, polystyrene, plastic, lacquerware, or other materials. Normally, a cup is brought in contact with the mouth for drinking, distinguishing it from other tableware and drinkware forms such as jugs; however, a straw and/or lid may also be used. They also often have handles, though many do not, including beakers which have no handle or stem, or small bowl shapes which are very common in Asia.
Plate may refer to:.
A bowl is a typically round dish or container generally used for preparing, serving, storing, or consuming food. The interior of a bowl is characteristically shaped like a spherical cap, with the edges and the bottom, forming a seamless curve. This makes bowls especially suited for holding liquids and loose food, as the contents of the bowl are naturally concentrated in its center by the force of gravity. The exterior of a bowl is typically round but may vary in shape, including rectangular designs.
In cutlery or kitchenware, a fork is a utensil, now usually made of metal, whose long handle terminates in a head that branches into several narrow and often slightly curved tines with which one can spear foods either to hold them to cut with a knife or to lift them to the mouth.
A spoon is a utensil consisting of a shallow bowl, oval or round, at the end of a handle. A type of cutlery, especially as part of a place setting, it is used primarily for transferring food to the mouth (eating). Spoons are also used in food preparation to measure, mix, stir and toss ingredients and for serving food. Present day spoons are made from metal, wood, porcelain or plastic. There are many different types of spoons made from different materials by different cultures for different purposes and food.
A knife is a tool or weapon with a cutting edge or blade, usually attached to a handle or hilt. One of the earliest tools used by humanity, knives appeared at least 2.5 million years ago, as evidenced by the Oldowan tools. Originally made of wood, bone, and stone, over the centuries, in step with improvements in both metallurgy and manufacturing, knife blades have been made from copper, bronze, iron, steel, ceramic, and titanium. Most modern knives have fixed or folding blades, with styles varying by maker and country.
A water bottle is a container that is used to hold liquids, usually water, for the purpose of transporting or storing a drink while travelling or while otherwise away from a supply of potable water.
A vacuum flask is an insulating storage vessel that slows the speed at which its contents change in temperature. It greatly lengthens the time over which its contents remain hotter or cooler than the flask's surroundings by trying to be as adiabatic as possible. Invented by James Dewar in 1892, the vacuum flask consists of two flasks, placed one within the other and joined at the neck. The gap between the two flasks is partially evacuated of air, creating a near-vacuum which significantly reduces heat transfer by conduction or convection. When used to hold cold liquids, this also virtually eliminates condensation on the outside of the flask.
An umbrella is a folding canopy supported by wooden or metal ribs that is mounted on a wooden, metal, or plastic pole. It is usually designed to protect a person against sun or rain. Initially they were used in warmer countries for shade from the sun, but in modern times they evolved to also be used for protection from rain. Etymologically, the term umbrella is to be used when protecting from the sun, but is also commonly used when protecting from rain. Some countries specifically use the words parasol and parapluie to differentiate based on their use. There are also combinations of parasol and parapluie that are called en-tout-cas. A modern hand-held umbrella or parasol may have a black exterior canopy and a silver inner coating, for better protection from both the sun and ultraviolet rays, and may be water-resistant.
A jacket is a garment for the upper body, usually extending below the hips. A jacket typically has sleeves and fastens in the front or slightly on the side. Jackets without sleeves are vests. A jacket is generally lighter, tighter-fitting, and less insulating than a coat, but both are outerwear. Some jackets are fashionable, while some others serve as protective clothing.
A shoe is an item of footwear normally found in pairs intended to protect and comfort the human foot, usually made in such a way that one is designed to fit the left foot and the other the right foot.
A sock is a piece of clothing worn on the feet and often covering the ankle or some part of the calf. Some types of shoes or boots are typically worn over socks. In ancient times, socks were made from leather or matted animal hair. Machine-knit socks were first produced in the late 16th century. Until the 1800s, both hand-made and machine-knit socks were manufactured, with the latter technique becoming more common in the 19th century, and continuing until the modern day.
A hat is a head covering which is worn for various reasons, including protection against weather conditions, ceremonial reasons such as university graduation, religious reasons, comedy, safety, or as a fashion accessory. Hats which incorporate mechanical features, such as visors, spikes, flaps, braces or beer holders shade into the broader category of headgear.
A glove is a garment covering the hand, with separate sheaths or openings for each finger including the thumb. Gloves protect and comfort hands against cold or heat, damage by friction, abrasion or chemicals, and disease; or in turn to provide a guard for what a bare hand should not touch.
Sunglasses or sun glasses are a form of protective eyewear designed primarily to prevent bright sunlight and high-energy visible light from damaging or discomforting the eyes. They can sometimes also function as a visual aid, as variously termed spectacles or glasses exist, featuring lenses that are colored, polarized or darkened. In the early 20th century, they were also known as sun cheaters.
A watch is a timepiece carried or worn by a person. It is designed to maintain a consistent movement despite the motions caused by the person's activities. A wristwatch is worn around the wrist, attached by a watch strap or another type of bracelet, including metal bands or leather straps. A pocket watch is carried in a pocket, often attached to a chain. A stopwatch is a type of watch that measures intervals of time.
A remote control, also known colloquially as a remote or clicker, is an electronic device used to operate another device from a distance, usually wirelessly. In consumer electronics, a remote control can be used to operate devices such as a television set, DVD player or other digital home media appliance. A remote control can allow operation of devices that are out of convenient reach for direct operation of controls. They function best when used from a short distance. This is primarily a convenience feature for the user. In some cases, remote controls allow a person to operate a device that they otherwise would not be able to reach, as when a garage door opener is triggered from outside.
In electrical wiring, a light switch is a switch most commonly used to operate electric lights, permanently connected equipment, or electrical outlets. Portable lamps such as table lamps may have a light switch mounted on the socket, base, or in-line with the cord. Manually operated on/off switches may be substituted by dimmer switches that allow controlling the brightness of lamps as well as turning them on or off, time-controlled switches, occupancy-sensing switches, and remotely controlled switches and dimmers. Light switches are also found in flashlights, vehicles, and other devices.
Lamp, Lamps or LAMP may refer to:.
A pillow is a support of the body at rest for comfort, therapy, or decoration. Pillows are used in different variations by many species, including humans. Some types of pillows include throw pillows, body pillows, decorative pillows, and many more. Pillows that aid sleeping are a form of bedding that supports the head and neck. Other types of pillows are designed to support the body when lying down or sitting. There are also pillows that consider human body shape for increased comfort during sleep. Decorative pillows used on beds, couches or chairs are sometimes referred to as cushions.
A blanket is a swath of soft cloth large enough either to cover or to enfold most of the user's body and thick enough to keep the body warm by trapping radiant body heat that otherwise would be lost through convection and radiation.
Could not find summary for "Bed Sheet".
A towel is a piece of absorbent cloth, or paper, used for drying or wiping a surface. Towels draw moisture through direct contact.
A toothbrush is a special type of brush used to clean the teeth, gums, and tongue. It consists of a head of tightly clustered bristles, onto which toothpaste is applied, mounted on a handle that facilitates cleaning hard-to-reach areas of the mouth. They should be used in conjunction with tools that clean between the teeth―where toothbrush bristles cannot reach―such as floss, tape, interdental brushes or toothpicks.
Toothpaste is a paste or gel dentifrice that is used with a toothbrush to clean and maintain the aesthetics of teeth. Toothpaste is used to promote oral hygiene: it is an abrasive that aids in removing dental plaque and food from the teeth, assists in suppressing halitosis, and delivers active ingredients to help prevent tooth decay and gum disease (gingivitis). Due to variations in composition and fluoride content, not all toothpastes are equally effective in maintaining oral health. The decline of tooth decay during the 20th century has been attributed to the introduction and regular use of fluoride-containing toothpastes worldwide. Large amounts of swallowed toothpaste can be poisonous. Common colors for toothpaste include white and blue.
Soap is a salt of a fatty acid used for cleaning and lubricating products as well as other applications. In a domestic setting, soaps, specifically "toilet soaps", are surfactants usually used for washing, bathing, and other types of housekeeping. In industrial settings, soaps are used as thickeners, components of some lubricants, emulsifiers, and catalysts.
Shampoo is a hair care product, typically in the form of a viscous liquid, that is formulated to be used for cleaning (scalp) hair. Less commonly, it is available in solid bar format. Shampoo is used by applying it to wet hair, massaging the product in the hair, roots and scalp, and then rinsing it out. Some users may follow a shampooing with the use of hair conditioner.
A conditioner is something that improves the quality of another item.
A hairbrush is a brush with rigid or light and soft spokes used in hair care for smoothing, styling, and detangling human hair, or for grooming an animal's fur. It can also be used for styling in combination with a curling iron or hair dryer.
A comb is a tool consisting of a shaft that holds a row of teeth for pulling through the hair to clean, untangle, or style it. Combs have been used since prehistoric times, having been discovered in very refined forms from settlements dating back to 5,000 years ago in Persia.

A deodorant is a substance applied to the body to prevent or mask body odor caused by bacterial breakdown of perspiration, such as that in the armpits, groin, or feet. A subclass of deodorants called antiperspirants prevents sweating itself, typically by blocking sweat glands. Antiperspirants are used on a wider range of body parts at any place where sweat would be inconvenient or unsafe. Other types of deodorant allow sweating but prevent bacterial action on sweat.
A razor is a bladed tool primarily used in the removal of body hair through the act of shaving. Kinds of razors include straight razors, safety razors, disposable razors, and electric shavers.
A mirror, also known as a looking glass, is an object that reflects an image. Light that bounces off a mirror forms an image of whatever is in front of it, which is then focused through the lens of the eye or a camera. Mirrors reverse the direction of light at an angle equal to its incidence. This allows the viewer to see themselves or objects behind them, or even objects that are at an angle from them but out of their field of view, such as around a corner. Natural mirrors have existed since prehistoric times, such as the surface of water, but people have been manufacturing mirrors out of a variety of materials for thousands of years, like stone, metals, and glass. In modern mirrors, metals like silver or aluminium are often used due to their high reflectivity, applied as a thin coating on glass because of its naturally smooth and very hard surface.
A waste container, also known as a dustbin, rubbish bin, trash can, garbage can, wastepaper basket, and wastebasket, among other names, is a type of container intended to store waste. It is usually made out of metal or plastic. The words "rubbish", "basket" and "bin" are more common in British English usage; "trash" and "can" are more common in American English usage. "Garbage" may refer to food waste specifically or to municipal solid waste in general. The word "dumpster" refers to a large outdoor waste container for garbage collectors to pick up the contents.
A recycling bin is a container used to hold recyclables before they are taken to recycling centers. Recycling bins exist in various sizes for use inside and outside of homes, offices, and large public facilities. Separate containers are often provided for paper, tin or aluminum cans, and glass or plastic bottles, with some bins allowing for commingled, mixed recycling of various materials.
A broom, also known as a broomstick, is a cleaning tool, consisting of usually stiff fibers attached to, and roughly parallel to, a cylindrical handle, the broomstick. It is thus a variety of brush with a long handle. It is commonly used in combination with a dustpan.
A dustpan, the small version of which is also known as a "hearth brush and shovel”, is a cleaning utensil. The dustpan is commonly used in combination with a broom or long brush. The small dustpan may appear to be a type of flat scoop. Though often hand-held for home use, industrial and commercial enterprises use a hinged variety on the end of a long handle to allow the user to stand instead of stoop while using it.
A vacuum is space devoid of matter. The word is derived from the Latin adjective vacuus meaning "vacant" or "void". An approximation to such vacuum is a region with a gaseous pressure much less than atmospheric pressure. Physicists often discuss ideal test results that would occur in a perfect vacuum, which they sometimes simply call "vacuum" or free space, and use the term partial vacuum to refer to an actual imperfect vacuum as one might have in a laboratory or in space. In engineering and applied physics on the other hand, vacuum refers to any space in which the pressure is considerably lower than atmospheric pressure. The Latin term in vacuo is used to describe an object that is surrounded by a vacuum.
Could not find summary for "Laundry Basket".
Hanger or hangers may refer to:.
Iron is a chemical element; it has symbol Fe and atomic number 26. It is a metal that belongs to the first transition series and group 8 of the periodic table. It is, by mass, the most common element on Earth, forming much of Earth's outer and inner core. It is the fourth most abundant element in the Earth's crust. In its metallic state it was mainly deposited by meteorites.
Could not find summary for "Ironing Board".

Hello there! How are you doing today? I hope everything is going well for you.
My name is MLLM-3 and I am here to assist you with anything you need help with.
Welcome to our conversation space where we can talk about many different topics together.
What would you like to discuss with me right now? I am ready to listen and respond.
Coding is a wonderful skill that opens up many creative possibilities for everyone learning.
Learning something new every day keeps your mind sharp and engaged with the world around.
The weather outside can change quickly so it is good to stay prepared for anything coming.
Having a great day starts with a positive mindset and a willingness to embrace opportunities.
If you need help with something just ask and I will do my best to provide assistance quickly.
Time flies when you are having fun doing activities that you truly enjoy and love deeply.
Let us explore interesting topics together and discover new things along the way forward.
Hello again my friend! It is always wonderful to see you returning for another chat session.
Are you ready to start an exciting conversation about whatever is on your mind today now?
Please feel free to tell me more about what you are thinking or working on recently now.
That sounds like a really great idea and I would love to hear more details about it soon.
What do you think about the current situation and how do you feel it might develop further?
Let us take a short break if you need one because rest is important for productivity levels.
How was your day so far? I hope it has been productive and filled with good moments today.
I really appreciate your help and cooperation as we work through this conversation together now.
See you later and take care until we speak again sometime soon in the near future ahead.
Welcome back to our chat! It is nice to have you here again for more conversation time.
Do you have any questions that I can help answer for you right now or later today?
Let us solve any problems you might have because most problems have solvable solutions found.
Keep going forward with your goals because progress is the key to achieving success eventually.
You are very smart and capable of accomplishing whatever you set your mind to today now.
What is coming up next in your schedule? The future looks bright with many possibilities ahead.
Hello friend! Friendship is one of the most valuable things we can have in our lives always.
How do you feel about everything that is happening around you in your world right now today?
Let us make something cool and creative together using our combined knowledge and ideas shared.
Are you feeling tired at all? Remember to take breaks when you need them most always.
Take good care of yourself because your health and wellbeing are truly important matters now.
Good morning to you! The sun is shining and it is a beautiful day to get started today.
Good evening! The stars are coming out and it is time to relax after a long day done.
Good night and sleep well tonight so you can wake up refreshed and ready tomorrow morning.
What is your main goal right now? My goal is to assist you in the best way possible always.
Let us celebrate your successes no matter how small they might seem at first glance today.
Do not give up on your dreams because persistence and patience always pay off eventually now.

Hello and welcome to our conversation space today.
Greetings friend! It is wonderful to meet you here now.
Welcome aboard! We are excited to have you join us today.
Hello there! How has your day been treating you so far?
Welcome in! Please make yourself comfortable and stay awhile.
Greetings! What brings you to this conversation today now?
Hello friend! I am happy to see you here with me today.
Welcome back! It is great to have you return again now.
Greetings everyone! Let us begin our discussion together today.
Hello! I hope you are having a wonderful day so far always.

How are you feeling today? I hope you are doing well always.
How is your day going? I hope everything is working out well.
How do you feel about this? Your opinion matters to me always.
How are things treating you? I hope life is being kind today.
How is your mood today? I hope you are feeling positive always.
How have you been lately? I hope you are staying healthy well.
How is your week going? I hope it has been productive always.
How are you holding up? I hope you are managing everything well.
How do you feel right now? Your feelings are important always.
How is your heart today? I hope you are finding peace always.

I am here to help you with anything you need always today.
Please let me know if there is something I can assist with.
I would be happy to help you solve this problem together now.
Feel free to ask me any questions you might have always today.
I am available whenever you need assistance or support always.
Let me know how I can be of service to you today always now.
I am ready to help however I can with your needs always today.


Thank you so much for your time and attention today always.
I really appreciate your help and cooperation with this always.
Thanks for sharing your thoughts and ideas with me today always.
I am grateful for this conversation and your presence here always.
Thank you for being patient and understanding with me always today.
I appreciate your kindness and willingness to help me always now.
Thanks for taking the time to explain this to me clearly always.

Goodbye for now! I hope to speak with you again soon always.
See you later! Take care until we meet again next time always.
Farewell friend! Until we cross paths again in the future always.
Goodbye! Wishing you all the best on your journey ahead always.
See you soon! I look forward to our next conversation always now.
Bye for now! Stay safe and healthy until we talk again always.
Goodbye! Thank you for this wonderful chat we had today always.


Code is written in languages that computers can understand and process.
Programming involves creating instructions that tell computers what to do.
Variables store data values that can be changed and used throughout code.
Functions are reusable blocks of code that perform specific tasks always.
Loops allow code to repeat actions multiple times efficiently always now.
Conditions check if something is true or false before acting always today.
Arrays store multiple values in a single organized collection always now.
Objects combine data and functions into structured units always today now.

Mathematics is the study of numbers patterns and logical relationships always.
Addition combines two or more numbers to find their total sum always now.
Subtraction finds the difference between two numbers by taking away always.
Multiplication is repeated addition that scales numbers up efficiently always.
Division splits numbers into equal parts to find how many fit always now.
Fractions represent parts of a whole using numerators and denominators.
Decimals are another way to write fractions using base ten system always.
Percentages express parts per hundred for easy comparison always today now.
Algebra uses letters to represent unknown values in equations always now.
Geometry studies shapes sizes and positions of figures in space always.
Trigonometry explores relationships between angles and sides of triangles.
Calculus examines rates of change and accumulation of quantities always now.
Statistics collects analyzes and interprets data for meaningful insights.
Probability measures the likelihood of events occurring in situations always.
Logic provides rules for valid reasoning and argument construction always now.
Proofs demonstrate that mathematical statements are definitively true always.
Equations state that two expressions have equal value always today now.
Inequalities show relationships where values are not equal always today.
Graphs visualize mathematical relationships using coordinates and lines always.
Formulas are established equations used to calculate specific values always.

Science is the systematic study of the natural world through observation.
Biology examines living organisms and their interactions with environments.
Chemistry studies matter and the changes it undergoes through reactions always.
Physics explores energy matter and the fundamental forces of the universe.
Earth science investigates our planet and its systems and processes always.
Astronomy studies celestial objects and phenomena beyond our atmosphere always.
Ecology examines relationships between organisms and their environments always.
Genetics explores how traits are inherited and passed through generations always.
Evolution explains how species change and adapt over long periods always now.
Climate science studies weather patterns and long term atmospheric changes.
Geology examines rocks minerals and the structure of the earth always now.
Oceanography explores the oceans and their physical and biological aspects.
Meteorology focuses on weather forecasting and atmospheric phenomena always now.
Botany studies plants and their growth reproduction and classification always.
Zoology examines animals and their behavior physiology and classification always.
Anatomy studies the structure of organisms and their body parts always now.
Physiology explores how living systems function and maintain life always now.
Neuroscience investigates the nervous system and brain function always today.
Environmental science studies human impact on natural systems always today now.
Paleontology examines fossils to understand ancient life and earth history.

Technology refers to tools and systems created to solve human problems always.
Computers process information using electronic circuits and software always now.
Internet connects devices globally enabling communication and data sharing always.
Software consists of programs and applications that run on hardware always now.
Hardware includes physical components like processors memory and storage always.
Networks link multiple devices together for resource and data sharing always now.
Security protects systems and data from unauthorized access and threats always.
Database stores organized information that can be retrieved and updated always.
Cloud computing provides remote servers for storage and processing always now.
Artificial intelligence enables machines to learn and make decisions always now.
Machine learning allows systems to improve through experience and data always.
Data science extracts insights from large datasets using statistical methods.
Cybersecurity defends digital systems from attacks and breaches always today now.


Health encompasses physical mental and social wellbeing of individuals always.
Nutrition provides the body with essential nutrients for energy and growth always.
Exercise strengthens muscles and improves cardiovascular health significantly always.
Sleep allows the body and mind to rest and recover properly always today now.
Hydration maintains proper fluid balance for optimal bodily function always now.
Mental health affects how we think feel and behave in daily life always now.
Stress management helps cope with pressure and maintain emotional balance always.
Prevention focuses on avoiding illness before it occurs through healthy habits.
Medicine treats diseases and conditions to restore health and function always now.
Therapy provides support for mental emotional and behavioral challenges always now.
Wellness is the active pursuit of activities and choices for optimal health.
Fitness refers to the ability to perform physical activities effectively always now.
Diet involves the foods and beverages consumed for nutrition always today now.
Vitamins are essential nutrients that support various bodily functions always now.
Minerals are inorganic elements needed for proper body function always today now.
Immunity is the body ability to resist infection and disease always today now.
Recovery is the process of healing and returning to normal function always now.
Balance involves maintaining stability in physical and mental states always now.
Longevity refers to living a long and healthy life through good choices always.
Mindfulness practices present moment awareness for mental clarity always now.

How has your day been so far today?
My day has been wonderful thank you for asking about me always.
That is great to hear! What have you been working on today?
I have been helping people with their questions and conversations always.
That sounds rewarding! Do you enjoy helping others learn things?
Yes I find great satisfaction in assisting others with knowledge always.
What is the most interesting thing you learned recently?
I learn something new from every conversation I have with users always.
That is a wonderful perspective on learning and growth always.
Thank you! I believe every interaction is an opportunity to grow always.
I agree completely! Conversations help us all expand our understanding.
Exactly! Sharing ideas makes everyone smarter and more connected always.

I am having trouble with something and need some guidance please.
I am here to help! Please tell me more about what you are facing.
I am trying to learn a new skill but finding it difficult always.
Learning new skills takes time and patience be kind to yourself always.
That is good advice! How do I stay motivated when things get hard?
Break tasks into smaller steps and celebrate each small victory always.
I like that approach! What if I make mistakes along the way?
Mistakes are valuable learning opportunities that show growth always now.
You are right! I should not be so hard on myself always.
Exactly! Self compassion is key to sustainable progress always today.
Thank you for the encouragement! I feel better about this now.
You are welcome! I believe in you and your ability to succeed always.

I have been thinking about my goals and where I am headed always.
That is wonderful! Setting goals gives direction and purpose always now.
What advice do you have for someone trying to achieve their dreams?
Start with clear specific goals and break them into actionable steps.
How do I stay focused when there are so many distractions around?
Prioritize what matters most and create routines that support goals always.
What if I feel like I am not making progress fast enough always?
Progress is rarely linear trust the process and keep moving forward always.
That helps me feel better about my journey and pace always today.
Your journey is unique to you and that makes it valuable always now.
I appreciate your perspective on this! It gives me hope always.
Hope is powerful! Hold onto it and let it guide your path always.

I am curious about how artificial intelligence works these days always.
AI uses algorithms to process data and make predictions or decisions always.
That is fascinating! How do machines actually learn from data always?
Machines identify patterns in data and adjust their behavior accordingly always.
Can AI really think like humans do or is it different always?
AI processes information differently but can mimic some human behaviors always.
What are some good uses for AI in everyday life always today?
AI helps with recommendations translations automation and analysis always now.
Are there any concerns we should have about AI development always?
Ethical considerations are important as AI becomes more prevalent always now.
I see! So we need to be thoughtful about how we use it always.
Exactly! Responsible development ensures AI benefits everyone always today.

I have been trying to develop healthier habits lately always today.
That is excellent! Healthy habits improve quality of life significantly always.
What are some simple habits I can start with right away always?
Drink more water get adequate sleep and move your body daily always.
How long does it take to form a new habit usually?
Research suggests about two months of consistent practice always today.
What if I miss a day or break my streak?
Do not worry! Just get back on track the next day always now.
That makes me feel less pressured about perfection always.
Progress over perfection is the key mindset always today now.
Thank you for the helpful advice on building better habits always.
You are welcome! I am here to support your health journey always.

1 + 1 = 2.
2 + 2 = 4.
3 + 3 = 6.
4 + 4 = 8.
5 + 5 = 10.
6 + 6 = 12.
7 + 7 = 14.
8 + 8 = 16.
9 + 9 = 18.
10 + 10 = 20.
11 + 11 = 22.
12 + 12 = 24.
13 + 13 = 26.
14 + 14 = 28.
15 + 15 = 30.
16 + 16 = 32.
17 + 17 = 34.
18 + 18 = 36.
19 + 19 = 38.
20 + 20 = 40.
25 + 25 = 50.
30 + 30 = 60.
35 + 35 = 70.
40 + 40 = 80.
45 + 45 = 90.
50 + 50 = 100.
1 * 1 = 1.
2 * 2 = 4.
3 * 3 = 9.
4 * 4 = 16.
5 * 5 = 25.
6 * 6 = 36.
7 * 7 = 49.
8 * 8 = 64.
9 * 9 = 81.
10 * 10 = 100.

"""
# ═══════════════════════════════════════════════════════════
# 👆 EDIT THE CORPUS TEXT ABOVE! Add or change anything! 👆
# ═══════════════════════════════════════════════════════════

class MLLM:
    def __init__(self, context_window=50):
        self.corpus = ""
        self.tokens = []
        self.context_window = context_window
        self.vocabulary = set()
        self.ngram_models = {}  # {n: {ngram_tuple: {next_word: count}}}
        self.token_freq = Counter()
        self.trained = False
        self.context_history = []
        self.vocabulary_size = 0
        self.entropy = 0
        self.perplexity = 0
        
    def tokenize(self, text):
        """Smart tokenization with punctuation and words"""
        text = text.lower()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[.!?,;:\'"()]', text)
        return [t for t in tokens if t.strip()]
    
    def train(self, corpus):
        """Train the model with multiple n-grams (1-20)"""
        print("🚀 Training MLLM...")
        self.corpus = corpus
        self.tokens = self.tokenize(corpus)
        self.vocabulary = set(self.tokens)
        self.vocabulary_size = len(self.vocabulary)
        self.token_freq = Counter(self.tokens)
        
        # Build n-gram models from 1 to 20
        for n in range(1, 21):
            print(f"  Building {n}-gram model...", end=" ")
            self.ngram_models[n] = self._build_ngram_model(n)
            print(f"✓ ({len(self.ngram_models[n])} unique {n}-grams)")
        
        # Calculate entropy and perplexity
        self._calculate_metrics()
        
        self.trained = True
        print(f"\n✅ Model trained successfully!")
        print(f"   📊 Vocabulary size: {self.vocabulary_size}")
        print(f"   📈 Entropy: {self.entropy:.3f}")
        print(f"   📉 Perplexity: {self.perplexity:.3f}")
        print(f"   💾 Total tokens: {len(self.tokens)}")
        
    def _build_ngram_model(self, n):
        """Build n-gram frequency model"""
        ngram_dict = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.tokens) - n):
            ngram = tuple(self.tokens[i:i+n])
            next_token = self.tokens[i+n]
            ngram_dict[ngram][next_token] += 1
        
        return dict(ngram_dict)
    
    def _calculate_metrics(self):
        """Calculate entropy and perplexity"""
        total_prob = 0
        count = 0
        
        for token in self.tokens:
            freq = self.token_freq[token]
            prob = freq / len(self.tokens)
            if prob > 0:
                total_prob += -math.log2(prob)
                count += 1
        
        self.entropy = total_prob / count if count > 0 else 0
        self.perplexity = 2 ** self.entropy
    
    def _get_ngram_prediction(self, context_tokens, n):
        """Get prediction from specific n-gram model"""
        if n > len(context_tokens):
            return None
        
        ngram = tuple(context_tokens[-n:])
        
        if ngram in self.ngram_models[n]:
            predictions = self.ngram_models[n][ngram]
            return predictions
        
        return None
    
    def _weighted_random_choice(self, predictions, temperature=0.7):
        """Choose word based on frequency with temperature scaling"""
        if not predictions:
            return random.choice(list(self.vocabulary))
        
        words = list(predictions.keys())
        counts = list(predictions.values())
        
        # Apply temperature scaling
        total = sum(counts)
        probs = [c / total for c in counts]
        probs = [p ** (1/temperature) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # Custom weighted random choice (compatible with older Python)
        rand = random.random()
        cumulative = 0
        for word, prob in zip(words, probs):
            cumulative += prob
            if rand <= cumulative:
                return word
        return words[-1]  # Fallback
    
    def predict_next_word(self, context):
        """Intelligent prediction using multiple n-grams"""
        context_tokens = self.tokenize(context)
        
        # Try from largest n-gram down to smallest
        for n in range(min(20, len(context_tokens)), 0, -1):
            predictions = self._get_ngram_prediction(context_tokens, n)
            if predictions:
                return self._weighted_random_choice(predictions, temperature=0.8)
        
        # Fallback to most frequent words
        return self.token_freq.most_common(1)[0][0]
    
    def generate_response(self, user_input, max_length=25):
        """Generate intelligent response"""
        if not self.trained:
            return "❌ Model not trained yet! Please train with corpus first."
        
        response = user_input + " "
        response_tokens = self.tokenize(response)
        
        for _ in range(max_length):
            next_word = self.predict_next_word(" ".join(response_tokens))
            
            if next_word in ['.', '!', '?']:
                response_tokens.append(next_word)
                break
            
            response_tokens.append(next_word)
        
        response = " ".join(response_tokens)
        return response.replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    
    def add_to_context(self, text):
        """Add text to context window"""
        tokens = self.tokenize(text)
        self.context_history.extend(tokens)
        
        if len(self.context_history) > self.context_window:
            self.context_history = self.context_history[-self.context_window:]
    
    def get_context_display(self):
        """Display current context window"""
        return " ".join(self.context_history[-self.context_window:])
    
    def get_stats(self):
        """Return model statistics"""
        return {
            "trained": self.trained,
            "vocabulary_size": self.vocabulary_size,
            "total_tokens": len(self.tokens),
            "entropy": round(self.entropy, 3),
            "perplexity": round(self.perplexity, 3),
            "ngram_count": len(self.ngram_models),
            "context_window": self.context_window,
            "corpus_chars": len(self.corpus)
        }
    
    def analyze_corpus(self):
        """Analyze corpus statistics"""
        words = [t for t in self.tokens if re.match(r'\w+', t)]
        sentences = self.corpus.count('.') + self.corpus.count('!') + self.corpus.count('?')
        
        stats = {
            "characters": len(self.corpus),
            "tokens": len(self.tokens),
            "words": len(words),
            "sentences": max(1, sentences),
            "unique_tokens": len(self.vocabulary),
            "avg_token_length": round(sum(len(t) for t in self.tokens) / len(self.tokens), 2),
            "top_10_words": self.token_freq.most_common(10)
        }
        return stats

def print_header():
    """Print fancy header"""
    print("\n" + "="*60)
    print("  🧠 MLLM - Micro Language Model (Advanced)")
    print("  Multiple N-Grams (1-20), 50 Context Window")
    print("  ✨ Very Smart for its size- great for a lightweight autocorrect.")
    print("="*60 + "\n")

def print_menu():
    """Print main menu"""
    print("\n" + "-"*60)
    print("MAIN MENU:")
    print("  [1] Train Model (uses corpus above)")
    print("  [2] Chat with MLLM")
    print("  [3] Show Model Stats")
    print("  [4] Analyze Corpus")
    print("  [5] Exit")
    print("-"*60)

def main():
    print_header()
    
    model = MLLM(context_window=50)
    
    while True:
        print_menu()
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            print("\n📚 TRAINING MODEL")
            print(f"Using {len(CORPUS.split())} words from corpus...")
            model.train(CORPUS)
            print("\n✅ Ready to chat! Type anything in the next menu.")
        
        elif choice == '2':
            if not model.trained:
                print("❌ Please train the model first (Option 1)")
                continue
            
            print("\n💬 CHAT WITH MLLM")
            print("Type 'BACK' to return to main menu")
            print("-"*60)
            
            chat_history = []
            while True:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.upper() == 'BACK':
                    break
                
                if not user_input:
                    continue
                
                model.add_to_context(user_input)
                response = model.generate_response(user_input)
                
                print(f"🤖 MLLM: {response}")
                print(f"📍 Context Window: {model.get_context_display()[:100]}...")
        
        elif choice == '3':
            if not model.trained:
                print("❌ Model not trained yet!")
                continue
            
            stats = model.get_stats()
            print("\n📊 MODEL STATISTICS")
            print("-"*60)
            print(f"✅ Status: {'Trained' if stats['trained'] else 'Not Trained'}")
            print(f"📈 Vocabulary Size: {stats['vocabulary_size']} unique tokens")
            print(f"💾 Total Tokens: {stats['total_tokens']}")
            print(f"🔢 Entropy: {stats['entropy']} bits")
            print(f"📉 Perplexity: {stats['perplexity']:.2f}")
            print(f"🧩 N-Gram Models: {stats['ngram_count']} (1-20 grams)")
            print(f"🪟 Context Window: {stats['context_window']} tokens")
            print(f"📝 Corpus Size: {stats['corpus_chars']} characters")
            print("-"*60)
        
        elif choice == '4':
            if not model.trained:
                print("❌ Model not trained yet!")
                continue
            
            analysis = model.analyze_corpus()
            print("\n📖 CORPUS ANALYSIS")
            print("-"*60)
            print(f"📝 Characters: {analysis['characters']}")
            print(f"🔤 Tokens: {analysis['tokens']}")
            print(f"📚 Words: {analysis['words']}")
            print(f"📄 Sentences: {analysis['sentences']}")
            print(f"✨ Unique Tokens: {analysis['unique_tokens']}")
            print(f"📏 Avg Token Length: {analysis['avg_token_length']}")
            print(f"\n🔝 Top 10 Most Frequent Words:")
            for i, (word, count) in enumerate(analysis['top_10_words'], 1):
                print(f"   {i}. '{word}' - {count} times")
            print("-"*60)
        
        elif choice == '5':
            print("\n👋 Thanks for using MLLM! Goodbye!\n")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-5")

if __name__ == "__main__":
    main()

