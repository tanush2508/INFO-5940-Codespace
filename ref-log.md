# Reflection – Multi-Agent Travel Planner

### What I Learned
Working on this project helped me understand how multi-agent systems can simulate teamwork between distinct roles. Designing the Planner and Reviewer felt like coordinating two different personalities — one creative and spontaneous, the other practical and detail-oriented. I learned how to structure prompts so that the Planner could generate rich, story-like itineraries while the Reviewer acted as a grounded fact-checker using the internet tool. It was fascinating to see how combining these agents produced more realistic and useful travel plans than either could alone.

### Challenges Faced
The biggest technical challenge was managing async behavior in Streamlit. I had to replace direct `asyncio.run()` calls with a safe wrapper to avoid event loop errors. Another challenge was finding the right prompt balance — at first, the Planner over-explained or made repetitive suggestions, and the Reviewer was too harsh. I tuned their tone and focus so that one sounded like an experienced travel blogger, and the other like a friendly but detail-oriented reviewer. I also spent time improving the UI: making tool logs collapsible, adding progress feedback, and writing softer error messages made the whole experience more enjoyable.

### Creative Design Choices
I decided to give the Planner a warm, narrative tone that makes the itinerary feel like it’s coming from a person who truly loves travel. The Reviewer, on the other hand, feels more analytical — almost like a travel editor ensuring everything makes sense. Together, they balance imagination and feasibility. These personas made the workflow feel human, not mechanical.

### Use of Tools or Assistance
I used ChatGPT (GPT-5) to refine prompts, structure the multi-agent flow, and debug async issues. All the creative direction and final design decisions were my own.


