---
title: Ask AI Makerspace
emoji: 🤖
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
---

<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

# Ask AI Makerspace

Huggingface space found: [here](https://huggingface.co/spaces/JKilpatrick/youtube_ai_makerspace)

## Overview
Welcome to Ask AI Makerspace - a communtiy-driven, open-source project aimed at developing a comprehensive Retrieval-Augmented Generation (RAG) application capable of answering questions related to AI Makerspace educational videos and documents. Inteded to be a learning space to get hands-on experience with LLMs and contributing to open-source projects.

Currently Works by pre-processing YouTube videos and retrieving information from the transcripts.
![youtube_rag_diagram](images/youtube_rag_diagram.png)
## How to Contribute
Welcoming contributions from anyone in the community heres how to contribute:
- **Submit a Pull Request:** Enhancements, bug fixes, documentation improvements.
- **Issue Tracking:** Report bugs, propose new features.

## Roadmap
- [ ] Clean up code and documentation.
     - Started as hackathon project so lots of messy boiler plate code that can use refining.
- [ ] Improve on existing system 
     -prompt engineering, chunking strategy, automatically indexing new videos, etc.
- [ ] Intergrate public github [repos](https://github.com/AI-Maker-Space/Awesome-AIM-Index).
     - Most videos have a corresponding github repo with the demo code. Would be awesome to have both elements incorporated combining conceptual ideas(YouTUbe) with code implementations(Github).
- [ ] Develop front-end to move off prtototype tools like huggingface spaces + chainlit and deploy to [aimakerspace.io](https://aimakerspace.io/).
     - more of a long shot goal and priority will be things related and taught in the courses as thats the fun stuff anyways.