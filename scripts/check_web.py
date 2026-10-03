import asyncio

from app.services.curio import CurioService
from app.tools.browser import UnavailableBrowser
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

async def main():
    curio = CurioService()

    try:
        print(
            "Active browser:",
            type(curio.browser).__name__,
        )

        if isinstance(
            curio.browser,
            UnavailableBrowser,
        ):
            raise SystemExit(
                "❌ Web browser is unavailable."
            )

        result = await curio.browser.search(
            "Maisammaguda Hyderabad spicy chicken biryani Zomato",
            5,
        )

        print("\nSearch result:")
        print(result)

        results = result.get(
            "results",
            []
        )

        if not results:
            raise SystemExit(
                "❌ Browser active but search returned no results."
            )

        print(
            "\n✅ Real web search is working."
        )

    finally:
        await curio.close()


if __name__ == "__main__":
    asyncio.run(main())