
import asyncio


import aiogram
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

import os

import logging
import bot_config
import bot_ai
import bot_conv_img
bot_ai.train()
bot = Bot(token = bot_config.TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def bot_start(message: Message):
    
    await message.answer('Привет!')

@dp.message(F.photo)
async def put_img(message: Message):
    await message.bot.download(file = message.photo[-1].file_id, destination = 'files\\test.jpg')
    await message.answer(str(bot_ai.test(bot_conv_img.conv('test.jpg'))))

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exit")










        







