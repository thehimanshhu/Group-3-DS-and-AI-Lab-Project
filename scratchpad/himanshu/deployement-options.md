# Deployment Options: Gradio App vs Chat Bot vs Both

**Status:** Rough draft for team discussion
**Related milestone:** Milestone 6 (Deployment & Documentation), due 10 Dec

---

## 1. Goal

Decide how users will access the trained model. The course requires deployment through one of:

- An API
- A demo interface
- Hugging Face Spaces

We compared three options: a **Gradio web app**, a **chat bot**, or **both**.

---

## 2. Option A: Gradio Web App (Hugging Face Spaces)

**How it works:** The user opens a web page, records or uploads audio, and sees the result.

**Pros:**

- Directly meets the course requirement (Hugging Face Spaces).
- Can show everything: verdict, confidence score and the **saliency plot** (which seconds look fake).
- Easy to build in Python, roughly 50–100 lines.
- Free hosting (CPU Basic: 2 vCPU, 16 GB RAM).
- Every Gradio Space gives a **free API** automatically, which other clients can call.
- Best for the live demo on the projector.

**Cons:**

- Users must open a website, which is less natural for checking a voice note they received.
- The free Space sleeps when idle, so the first request after a while is slow.
- Runs on CPU, so each clip takes a few seconds.

---

## 3. Option B: Chat Bot

### 3.1 WhatsApp (considered, dropped)

**Pros:**

- Where most suspicious voice notes in India actually arrive.

**Cons:**

- Needs a Meta developer account and business setup.
- The free test number can only message **5 verified phone numbers**.
- Temporary access tokens expire and must be renewed.
- The bot can only reply freely within 24 hours of the user's message.
- More setup work than the value it adds for a course demo.

### 3.2 Telegram (chosen bot platform)

**How it works:** The user forwards a voice note to the bot and gets a text reply, e.g. "⚠️ Likely AI-generated · 87%".

**Pros:**

- Free, with no business account and no limit on users.
- Setup in minutes with BotFather.
- Simple Python library (`python-telegram-bot`), roughly 30–60 lines.
- Natural for users: forward a voice note, get an answer.
- Good "wow" moment in the presentation.

**Cons:**

- Cannot show the saliency plot well (it could only be sent as an image).
- Needs a small server running to receive messages (webhook or polling).
- Voice notes are OGG/Opus and must be converted to 16 kHz WAV with ffmpeg.
- On its own, it does not clearly meet the "Hugging Face Spaces / demo interface" requirement.

---

## 4. Option C: Both (Recommended)

**How it works:** The model runs **once** inside the Gradio app on Hugging Face Spaces. The Telegram bot does not load the model; it sends the audio to the Gradio app's built-in API using `gradio_client`.

```
User (browser) ──► Gradio app on HF Spaces ──► XLS-R model ──► result
                          ▲
User (Telegram) ──► Telegram bot ──┘ (calls the Gradio API)
```

**Pros:**

- Meets the course requirement through the Gradio app.
- The bot adds real-world value with very little extra code.
- Only one model to maintain; no duplicate deployment.
- Two ways to show the project in the final presentation.
- If one fails during the demo, the other is a backup.

**Cons:**

- Two small pieces to keep running instead of one.
- The bot depends on the Space being awake.

---

## 5. Comparison

| Criteria | Gradio app | Telegram bot | Both |
| --- | --- | --- | --- |
| Meets course requirement | Yes | Weak | Yes |
| Shows saliency explanation | Yes | Limited | Yes |
| Easy for everyday users | Medium | Yes | Yes |
| Setup effort | Low | Low | Low–Medium |
| Cost | Free | Free | Free |
| Demo impact | Good | Very good | Best |

---

## 6. Decision

**Go with both, built in this order:**

1. **Gradio app first** (required): upload/record audio → verdict, confidence, saliency plot.
2. **Telegram bot second** (add-on): a thin client that calls the Gradio app's API.
3. **Backup:** run the Gradio app on Colab with `share=True` and keep a screen recording for the final presentation.

---

## 7. Open Questions

- [ ] Where should the bot run: a second Hugging Face Space, Render free tier, or a team member's laptop during the demo?
- [ ] Should the bot also send the saliency plot as an image?
- [ ] Who owns the bot (suggested: whoever finishes their milestone task early)?

---

## 8. Resources

- Gradio docs: <https://www.gradio.app/docs>
- Gradio on Hugging Face Spaces: <https://huggingface.co/docs/hub/spaces-sdks-gradio>
- Gradio Python client: <https://www.gradio.app/guides/getting-started-with-the-python-client>
- Telegram bot tutorial: <https://core.telegram.org/bots/tutorial>
- python-telegram-bot: <https://docs.python-telegram-bot.org>
- WhatsApp Cloud API (for reference): <https://developers.facebook.com/docs/whatsapp/cloud-api/get-started>
