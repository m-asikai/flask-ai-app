import { v4 as uuidv4 } from "https://jspm.dev/uuid";
import { marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";
import markedKatex from "https://esm.sh/marked-katex-extension@5";

marked.use(markedKatex({ throwOnError: false }));
const baseURl = "http://localhost:5000";
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send");
const discussion = document.getElementById("discussion");
const chatBox = document.getElementById("chatbox");
const courseInfo = document.getElementById("course-info");
const alphabetical = document.getElementById("alphabetic");
const byCredit = document.getElementById("by-credit");
const sessionId = uuidv4();
let data = null;
let originalData = null;
let defaultMessage = true;
let sorted,
  alpha,
  credit = false;

const init = async () => {
  await fetchData();
  showData();
};

const filter = () => {
  if (!alpha && !credit) {
    sorted = false;
  }
  if (alpha && credit) {
    sorted = true;
    data = originalData
      .toSorted((a, b) => a.name.localeCompare(b.name))
      .sort((a, b) => Number(b.credits) - Number(a.credits));
  } else {
    if (alpha) {
      sorted = true;
      data = originalData.toSorted((a, b) => a.name.localeCompare(b.name));
    }
    if (credit) {
      sorted = true;
      data = originalData.toSorted(
        (a, b) => Number(b.credits) - Number(a.credits)
      );
    }
  }

  courseInfo.textContent = "";
  showData();
};

alphabetical.addEventListener("click", () => {
  alpha = !alpha;
  alphabetical.style.backgroundColor = alpha ? "#b6b7bbff" : "#ffffff";
  filter();
});
byCredit.addEventListener("click", () => {
  credit = !credit;
  byCredit.style.backgroundColor = credit ? "#b6b7bbff" : "#ffffff";
  filter();
});

const updateConvo = (message) => {
  const styleClass = message.sender === "user" ? "user-message" : "answer";
  const div = document.createElement("div");
  div.setAttribute("class", styleClass);
  div.innerHTML += message.content;
  discussion.append(div);
  const scrollAmount =
    message.sender === "user"
      ? discussion.scrollHeight
      : discussion.scrollHeight - div.offsetHeight;
  discussion.scrollTo({
    top: scrollAmount,
    behavior: "smooth",
  });
};

const contactAI = async (messageToSend) => {
  let message = "";

  try {
    const bodyMessage = defaultMessage
      ? `There is a university course called ${messageToSend}, tell me about the topic and it's practical uses.
      Don't talk about how the course is executed or what happens in the course.`
      : messageToSend;
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: bodyMessage,
      }),
    };
    const res = await fetch(`${baseURl}/conv/${sessionId}`, options);
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    const preParsed = document.createElement("p");
    discussion.appendChild(preParsed);

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunkText = decoder.decode(value, { stream: true });
      message += chunkText;
      preParsed.textContent += chunkText;
    }
    discussion.removeChild(preParsed);
    updateConvo({
      sender: "ai",
      content: marked.parse(message),
    });
  } catch (err) {
    console.error("fetch error:", err);
    chatBox.textContent = `Error: ${err.message}`;
  }
};

const fetchData = async () => {
  if (!originalData) {
    let res = await fetch(`${baseURl}/courses`);
    originalData = await res.json();
  }
};

const showData = () => {
  if (!originalData) {
    throw new Error("Data not found.");
  }
  const scopeData = sorted ? data : originalData;
  for (let course of scopeData) {
    const card = document.createElement("div");
    card.setAttribute("class", "card");
    const info = `<p class="info"><b>${course.name}</b></br>
                  Id: ${course.course_id} </br>
                  Credits: ${course.credits}</p>`;
    card.innerHTML += info;
    const button = document.createElement("p");
    button.setAttribute("class", "button");
    button.textContent = "Learn more";
    button.addEventListener("click", () => {
      contactAI(course.name);
      defaultMessage = true;
      userInput.disabled = false;
      userInput.placeholder = "";
      chatBox.scrollIntoView({ behavior: "smooth" });
    });
    card.appendChild(button);
    courseInfo.appendChild(card);
  }
};

const handleInput = () => {
  const inputContent = userInput.value;
  defaultMessage = false;
  contactAI(inputContent);
  updateConvo({
    sender: "user",
    content: inputContent,
  });
  userInput.value = "";
};

sendButton.addEventListener("click", handleInput);

userInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    handleInput();
  }
});

init();
