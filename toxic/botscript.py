import discord
import pickle
from discord.ext import commands

# Load model and vectorizer
with open("toxic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    msg = message.content
    X = vectorizer.transform([msg])
    prediction = model.predict(X)

    if prediction[0] == 1:
        await message.channel.send(
            f"⚠️ @{message.author.mention}, that message seems toxic. Please keep it respectful."
        )
    elif msg.lower() in ["hello", "hi", "hey"]:
        await message.channel.send(
            f"👋 Hello @{message.author.name}!"
        )

    # Allow commands to work if you plan to add any
    await bot.process_commands(message)

# Replace 'your_token_here' with your actual token
bot.run("your token name")
