<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI Scheduling Assistant | Agentic AI Hackathon</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Inter', sans-serif;
      background: #f9f9fc;
      color: #222;
      line-height: 1.7;
      margin: 0;
      padding: 0;
    }
    header {
      background: #6c5ce7;
      color: white;
      padding: 3rem 2rem;
      text-align: center;
    }
    header h1 {
      margin: 0;
      font-size: 2.5rem;
    }
    header p {
      margin-top: 0.5rem;
      font-size: 1.2rem;
      opacity: 0.9;
    }
    section {
      max-width: 960px;
      margin: 2rem auto;
      padding: 0 1.5rem;
    }
    h2 {
      border-left: 5px solid #6c5ce7;
      padding-left: 1rem;
      color: #333;
    }
    h3 {
      color: #6c5ce7;
      margin-top: 1.5rem;
    }
    code {
      background: #eee;
      padding: 0.3rem 0.5rem;
      border-radius: 4px;
    }
    pre {
      background: #282c34;
      color: #f8f8f2;
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
    }
    ul {
      padding-left: 1.2rem;
    }
    a {
      color: #0984e3;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .badge {
      display: inline-block;
      background: #fd79a8;
      color: white;
      padding: 0.3rem 0.8rem;
      border-radius: 1rem;
      font-size: 0.85rem;
      margin-left: 0.5rem;
    }
    footer {
      text-align: center;
      padding: 2rem;
      font-size: 0.9rem;
      color: #777;
    }
  </style>
</head>
<body>

  <header>
    <h1>🤖 AI Scheduling Assistant</h1>
    <p>Agentic AI Hackathon | IIT Bombay</p>
  </header>

  <section>
    <h2>🚀 Introduction</h2>
    <p><strong>Problem Statement:</strong> Build an intelligent AI Scheduling Assistant that autonomously coordinates and optimizes meetings with minimal human input.</p>

    <h3>💡 Why Agentic AI?</h3>
    <ul>
      <li><strong>Reasoning:</strong> Resolves conflicts and prioritizes key attendees.</li>
      <li><strong>Acting:</strong> Sends follow-ups, reschedules intelligently.</li>
      <li><strong>Learning:</strong> Adapts to personal preferences over time.</li>
    </ul>
  </section>

  <section>
    <h2>✨ Key Features</h2>
    <ul>
      <li>📅 Autonomous Coordination</li>
      <li>🔄 Dynamic Adaptability</li>
      <li>💬 Natural Language Input</li>
      <li>🔗 Google Calendar Sync</li>
    </ul>
  </section>

  <section>
    <h2>🧪 Success Metrics</h2>
    <ul>
      <li>✅ <strong>Autonomy:</strong> Minimal human intervention</li>
      <li>✅ <strong>Accuracy:</strong> No conflicts or overlaps</li>
      <li>✅ <strong>UX:</strong> Time-saving and intuitive interface</li>
    </ul>
  </section>

  <section>
    <h2>⚙️ Setup Instructions</h2>
    <h3>📦 Prerequisites</h3>
    <pre><code>git clone https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant.git
cp -r AI-Scheduling-Assistant/* ./</code></pre>

    <h3>🖥️ MI300 Instance Access</h3>
    <p>Refer to this image:</p>
    <img src="https://github.com/user-attachments/assets/3b9d68c7-f994-486b-8734-ff61648bb192" alt="MI300 Access" style="max-width: 100%; border-radius: 8px; margin: 1rem 0;">
  </section>

  <section>
    <h2>📆 Google Calendar Integration</h2>
    <p>Authenticate and pull events programmatically using provided tokens.</p>
    <p><a href="https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant/blob/main/Calendar_Event_Extraction.ipynb" target="_blank">View Notebook →</a></p>
  </section>

  <section>
    <h2>⚡ vLLM Server Setup</h2>

    <h3>DeepSeek 7B Chat</h3>
    <pre><code>HIP_VISIBLE_DEVICES=0 vllm serve /home/user/Models/deepseek-ai/deepseek-llm-7b-chat \
    --gpu-memory-utilization 0.9 \
    --swap-space 16 \
    --port 3000 ...
    </code></pre>

    <h3>LLaMA 3.1 8B Instruct</h3>
    <pre><code>HIP_VISIBLE_DEVICES=0 vllm serve /home/user/Models/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.3 \
    --port 4000 ...
    </code></pre>
  </section>

  <section>
    <h2>🤖 AI Agent Logic</h2>
    <p>Example Python code:</p>
    <pre><code>class AI_AGENT:
  def __init__(self, client, MODEL_PATH):
      self.base_url = BASE_URL
      self.model_path = MODEL_PATH

  def parse_email(self, email_text):
      response = client.chat.completions.create(
          model=self.model_path,
          messages=[{
              "role": "user",
              "content": f"You are an agent... Email: {email_text}"
          }]
      )
      return json.loads(response.choices[0].message.content)
    </code></pre>
    <p><a href="https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant/blob/main/Sample_AI_Agent.ipynb" target="_blank">Open Full Notebook →</a></p>
  </section>

  <section>
    <h2>📥 Input Format</h2>
    <pre><code>{
  "Request_id": "...",
  "From": "...",
  "EmailContent": "Hi team, let's meet on Thursday for 30 minutes..."
}</code></pre>

    <h3>📤 Output Format</h3>
    <pre><code>{
  "EventStart": "...",
  "EventEnd": "...",
  "Duration_mins": "30"
}</code></pre>
  </section>

  <section>
    <h2>📬 Submission</h2>
    <ul>
      <li>Function: <code>your_meeting_assistant()</code></li>
      <li>Returns: Processed JSON with schedule info</li>
      <li>Deadline: 2:00 PM on Hackathon Day</li>
    </ul>
    <p><a href="https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant/blob/main/Submission.ipynb" target="_blank">View Submission Notebook →</a></p>
  </section>

  <section>
    <h2>🏆 Evaluation Criteria</h2>
    <ul>
      <li>✅ Output Correctness</li>
      <li>⚡ Response Latency</li>
      <li>📁 Code & Repo Quality</li>
      <li>🎯 Creativity in Implementation</li>
    </ul>
  </section>

  <footer>
    <p>Built for the AMD AI Sprint Hackathon at IIT Bombay · 2025 🚀</p>
  </footer>
</body>
</html>
