# Non-Tech Guide — SEO Multi-Agent Tool

**Who this is for:** Business, marketing, and operations people  
**Language:** Simple English only

Engineers should also open `Technical.md` / `Technical.pdf`.

---

## 1. Why we built this (the real purpose)

### We are not here for likes and views

Many websites and social posts are written mainly to get:

- more views  
- more followers  
- more attention  

Those posts can look exciting. People click. People scroll. But after reading, the person still feels stuck.

**That is not our goal.**

### We are here to help people with real problems

We create content so readers can **learn something useful** and feel less stuck with a real issue — in any topic or domain.

If our content cannot fully solve their problem, it should still:

- point them clearly to **where to go next**, and  
- do that **without confusing** them  

**Simple rule:** help the reader → not chase attention.

### Example (easy to understand)

A person wants to learn **LangGraph** and **persistent memory**.

They search Google or social media. They find many popular posts with lots of views. Those posts may:

- sound exciting for a minute  
- use big words  
- skip the hard but important parts  

So the person feels interested… but still does not really understand, and still cannot fix their issue.

**Our articles and blogs should fill that gap.**

After reading our content, the person should either:

1. **Actually learn** something they can use, **or**  
2. Know the **clear next step** to learn it properly — without feeling lost  

That is why this whole workflow exists.

---

## 2. What is this tool?

It is a **content helper**.

Every morning (example: **9 AM**), an automation tool called **n8n** looks at a Google Sheet for topics that are still **pending**. It sends each topic to our **AI writing team**. The team writes the content, checks quality, and then a real person approves it by email. After approval, the sheet says **posted** and that row is finished.

Think of it like a magazine office: robots draft the article, a human editor clicks Approve — and every draft should still try to **help**, not just impress.

---

## 3. How you generate content (two easy ways)

You do **not** need to know coding. Pick the way that fits your day.

### Way 1 — Use the web interface (right now)

1. Open the app link you were given (the web page).  
2. Type **what you want to generate** (the topic or brief) in the box.  
3. Choose brand / content type if asked.  
4. Click the **Generate** button.  

That is it.

The full AI pipeline (research → plan → write → review) usually takes about **1–2 minutes**. When it finishes, the draft appears on the same page so you can read, copy, or download it.

**Tip:** Leave the page open while it runs. Do not close the tab until you see the result.

---

### Way 2 — Use the Google Sheet (daily at 9 AM)

You will get a **Google Sheet link**. Use it like this:

1. Open the sheet.  
2. In the **topic** column, write the topic name (what you want written).  
3. In the **status** column, put **Pending**.  

Example:

| topic | status |
|-------|--------|
| How to start digital marketing for a startup | Pending |

**What happens next (automatic):**

- Every day at **9 AM**, the system looks for rows with status **Pending**.  
- It runs the full writing pipeline for those topics.  
- When done, it updates the **generated_topic** (or content) column with the result.  
- You get an **alert for approval** (usually by email) so a human can say yes or no.

You do not need to click Generate for sheet topics — filling **topic** + **Pending** is enough. The 9 AM run does the rest.

---

## 4. Proper steps (start to finish)

Read this list in order. This is how the whole system works.

### Step A — Clock rings (example: 9:00 AM)

n8n starts automatically (or someone runs it by hand).

### Step B — Read Google Sheet

n8n opens the sheet and finds rows where:

- **Status** = `pending`

Example row:

| Topic | Brand | Status |
|-------|-------|--------|
| How to start digital marketing for a startup | Futuristix | pending |

### Step C — Call the AI writing API

n8n sends that topic to the **LangGraph API** (our backend).

This starts the **multi AI agent** pipeline.

### Step D — Shared notebook (important idea)

All agents share one notebook called **state** (`state.py` for engineers).

- Each agent **reads** what it needs from the notebook  
- Each agent **writes** its answer back into the same notebook  
- The next agent uses that answer  

Like one shared folder everyone updates in turn.

### Step E — The five agents (in this exact order)

LangGraph is the **traffic controller**. It decides who goes next (graphing + routing).

```
1. Manager   (manager.py)
2. Research  (research.py)
3. Strategy  (strategy.py)
4. Writer    (writer.py)
5. Review    (review.py)
     │
     └── if quality is low → back to Writer (max 3 times)
```

#### 1) Manager

- Checks the request  
- Picks the **brand** (example: Futuristix)  
- Understands if this should be article / email / social, etc.  
- Writes brand details into the shared notebook  

#### 2) Research

- Searches the web / news / videos for useful facts  
- Collects sources and number-style stats  
- Writes the research pack into the notebook  

#### 3) Strategy (three special jobs)

Strategy plans “how to write,” using three helpers:

**A. SEO helper (`seo_service`)**

- Creates keyword lists: **primary**, **secondary**, and **long-tail**  
- Uses the research results and computer “meaning scores”  
- Gives each keyword a score using things like:
  - how often it appears in research (**frequency**)  
  - how close it is to the topic meaning (**semantic similarity**)  
  - search **intent** (info vs buy vs browse)  
  - match to customer **pain points**  
  - brand / **authority** style fit  
- Then it **re-ranks** keywords (best ones go first)

**B. Hashtag helper**

- Suggests hashtags (mainly useful for social)

**C. Citation helper**

- Cleans up source names / links for the article

Strategy also builds the **outline** (heading plan) and the writing blueprint.

#### 4) Writer

- Reads the blueprint + research from the notebook  
- Writes the full content  
- Uses packaging helpers (**formatter**, **json_builder**) so the answer is neat for the sheet/API  
- Hands the draft to Review  

#### 5) Review

- Scores the content out of 100  
- **Target score = 95**  
- If score is below 95 → send notes back to Writer and try again  
- This Writer ↔ Review loop can run up to **3 times**  
- Then the pipeline stops and returns the best result it has  

### Step F — Back to n8n + Google Sheet update

When the AI team finishes, n8n:

1. Puts the full article into the sheet column **Content generated**  
2. Saves the score  
3. Changes status from **pending** → **pending review**  

### Step G — Human approval by email

n8n sends an **email alert** to a person:

- Topic  
- Score  
- Preview / link to content  
- Buttons or reply: **Approve** / **Reject**  

### Step H — What happens next?

#### If human Approves

1. Sheet status becomes **posted**  
2. This row’s whole loop **stops** ✔  

#### If human Rejects

1. Content must be **created again** (n8n calls the AI API again)  
2. Inside the AI team, Writer ↔ Review can still loop up to **3 times**  
3. If after recreates / retries it is still not good enough → **alert a developer ASAP** so they can check and fix the system  

---

## 5. Status meanings (sheet)

| Status | Simple meaning |
|--------|----------------|
| **Pending** (or pending) | Waiting. Not written yet. Put this after you add a **topic**. |
| **pending review** | Written. Waiting for human yes/no. Result is in **generated_topic** / content. |
| **posted** | Approved. Finished. |
| **rejected** | Human said no. Will recreate or need developer help. |

Flow:

```
pending
   → pending review   (content filled + email sent)
        → posted                 (approve → stop)
        → recreate → try again   (reject)
             → still bad → alert developer
```

---

## 6. Simple architecture picture

```
9 AM clock
    |
    v
[ n8n ] ---- reads ----> [ Google Sheet: pending topics ]
    |
    | calls API
    v
[ LangGraph traffic controller ]
    |
    +--> Manager
    +--> Research
    +--> Strategy (SEO + Hashtags + Citations)
    +--> Writer (+ formatter / json packer)
    +--> Review  --(if <95)--> Writer again (max 3)
    |
    v
[ n8n ] ---- writes content + "pending review" ----> [ Google Sheet ]
    |
    v
[ Email to human ]
    |
    +-- Approve --> status = posted --> STOP
    +-- Reject  --> recreate (up to 3) --> else ALERT DEVELOPER
```

---

## 7. Shared notebook example

Imagine the notebook after each agent:

1. **Manager adds:** brand = Futuristix, tone = practical  
2. **Research adds:** 10 articles + 5 stats  
3. **Strategy adds:** top keywords + outline + citations  
4. **Writer adds:** full article text  
5. **Review adds:** score = 96, status = pass  

Then n8n copies the article text into the sheet.

---

## 8. Brands (who we write as)

| Brand | Simple idea | Ending ask example |
|-------|-------------|--------------------|
| Futuristix | AI / growth for businesses | Book an AI Discovery Call |
| GTIB | Buying / selling companies | Schedule an Advisory Call |
| Kinvo | Childcare | Book a Consultation |
| MPM | Property | Contact Property Advisor |
| GCB | Engineering | Contact Our Engineering Team |

---

## 9. What content types exist?

| Type | Simple meaning |
|------|----------------|
| Article / Blog | Long guide or blog post |
| Email | Email campaign text |
| SEO content | Article plus keyword notes |
| Social | LinkedIn / short posts |

---

## 10. Quality score in plain words

Review checks:

| Check | Meaning |
|-------|---------|
| Content quality | Clear and useful? |
| SEO | Right search words? |
| Brand match | Sounds like our brand? |
| Structure | Good beginning / middle / end? |
| Facts | Claims backed by sources? |
| Call to action | Clear next step? |

Goal: **95+**. Retries inside AI: **up to 3**.

---

## 11. One full day example

**8:55 AM**  
Sheet row: topic ready, status = pending  

**9:00 AM**  
n8n starts → finds pending row → calls AI API  

**9:08 AM** (example timing)  
Manager → Research → Strategy (SEO/hashtags/citations) → Writer → Review  
(maybe Writer fixes twice if score was low)  

**9:10 AM**  
Sheet: Content generated filled, status = pending review  
Email: “Please approve Futuristix article (score 95)”  

**10:00 AM**  
You click Approve → status = posted → stop  

**If you Reject**  
AI writes again (up to 3 tries). If still wrong → developer gets an urgent alert.

---

## 12. Who does what?

| Job | Who |
|-----|-----|
| Add topics to sheet (**topic** + **Pending**) | Marketing / ops |
| Use web page (type topic → **Generate**) | Marketing / ops |
| Run at 9 AM, fill **generated_topic**, send approval alert | n8n |
| Research, plan, write, score | AI agents in this project |
| Approve / reject | Human |
| Fix system if rejects keep failing | Developer |

---

## 13. Common questions

**Q: Do I need to know code?**  
A: No. Use the web page (type topic → Generate), or the sheet (**topic** + **Pending**) and the approval alert.

**Q: How long does the web Generate button take?**  
A: Usually about **1–2 minutes** for the full pipeline.

**Q: Does “posted” mean it is live on the website?**  
A: In this design it means “approved and finished in the workflow.” Website publish can be a later extra step.

**Q: Why 9 AM?**  
A: Just an example schedule. Your team can choose any time in n8n.

**Q: What if the score is 90 but a human likes it?**  
A: Human can still approve. The 95 goal guides the AI; humans have final say.

**Q: How should we judge content when approving?**  
A: Ask: “Would this actually help someone stuck on this problem, or is it only flashy?” Prefer learning and clear next steps over views and hype.

---

## 14. One-page cheat sheet

**Purpose first:** help readers solve real problems — not chase views.

**On the web page:** write what you want → click **Generate** → wait about **1–2 minutes**.

**On the Google Sheet:** put the topic in **topic**, put **Pending** in **status** → daily **9 AM** run → **generated_topic** fills in → **alert for approval**.

1. **9 AM** — n8n starts  
2. Take **Pending** topics from Google Sheet  
3. Call LangGraph / multi-agent API  
4. **Manager → Research → Strategy → Writer → Review**  
5. Everyone uses shared **state** notebook  
6. Strategy = **SEO + hashtags + citations** (SEO re-ranks keywords)  
7. Writer packages with formatter / json builder  
8. Review loop max **3**, target score **95**  
9. Sheet gets **generated_topic** / content + status **pending review**  
10. Email / alert to human for approval  
11. Approve → **posted** → stop  
12. Reject → recreate (up to 3) → else **alert developer**

---

*Non-tech guide. Engineers: see Technical.md / Technical.pdf for file paths and scoring weights.*
