from google import genai

client = genai.Client(api_key="AIzaSyDJ4BITIcCnTNF4_OCp8rtwiBvYJvxvjaU")
cache = {}
def neutralize_text(text):

    # return cached response
    if text in cache:
        return cache[text]

    try:

        prompt = f"""
        Rewrite this text in a respectful and non-toxic way
        without changing the meaning.

        Text:
        {text}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        neutralized = response.text

        # store in cache
        cache[text] = neutralized

        return neutralized

    except Exception as e:

        print("Gemini Error:", e)

        return text

if __name__ == "__main__":

    text = "You are a stupid useless person"

    output = neutralize_text(text)

    print(output)