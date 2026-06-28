import feedparser
import pandas as pd
from newspaper import Article


def extract_article_content(url):

    try:

        article = Article(url)

        article.download()
        article.parse()

        content = article.text.strip()

        try:

            article.nlp()

            summary = (
                article.summary.strip()
                if article.summary
                else ""
            )

        except Exception:

            summary = ""

        # Fallback summary if NLP summary fails
        if not summary and content:

            summary = content[:500]

        return {
            "content": content,
            "summary": summary
        }

    except Exception as e:

        print(
            f"Article Extraction Error: {e}"
        )

        return {
            "content": "",
            "summary": ""
        }


def fetch_portfolio_news(selected_assets):
    """
    Fetch general Indian financial news and
    asset-specific news from Google News RSS.
    """

    news_items = []

    try:

        # General Indian market news
        general_url = (
            "https://news.google.com/rss/search?"
            "q=Indian+stock+market+OR+NSE+OR+BSE"
            "&hl=en-IN&gl=IN&ceid=IN:en"
        )

        general_feed = feedparser.parse(
            general_url
        )

        for entry in general_feed.entries[:10]:

            article_data = extract_article_content(
                entry.get(
                    "link",
                    ""
                )
            )

            news_items.append(
                {
                    "Headline": entry.get(
                        "title",
                        ""
                    ),
                    "Source": getattr(
                        entry,
                        "source",
                        {}
                    ).get(
                        "title",
                        "Google News"
                    ),
                    "Published": entry.get(
                        "published",
                        "N/A"
                    ),
                    "Link": entry.get(
                        "link",
                        ""
                    ),
                    "Summary": article_data[
                        "summary"
                    ],
                    "Content": article_data[
                        "content"
                    ]
                }
            )

        # Asset-specific news
        for ticker in selected_assets:

            asset = ticker.replace(
                ".NS",
                ""
            )

            asset_url = (
                f"https://news.google.com/rss/search?"
                f"q={asset}"
                f"&hl=en-IN&gl=IN&ceid=IN:en"
            )

            asset_feed = feedparser.parse(
                asset_url
            )

            for entry in asset_feed.entries[:3]:

                article_data = extract_article_content(
                    entry.get(
                        "link",
                        ""
                    )
                )

                news_items.append(
                    {
                        "Headline": entry.get(
                            "title",
                            ""
                        ),
                        "Source": getattr(
                            entry,
                            "source",
                            {}
                        ).get(
                            "title",
                            "Google News"
                        ),
                        "Published": entry.get(
                            "published",
                            "N/A"
                        ),
                        "Link": entry.get(
                            "link",
                            ""
                        ),
                        "Summary": article_data[
                            "summary"
                        ],
                        "Content": article_data[
                            "content"
                        ]
                    }
                )

        news_df = pd.DataFrame(
            news_items
        )

        print(
            f"Total News Items: {len(news_items)}"
        )

        print(
            news_df.head()
        )

        if news_df.empty:

            return news_df

        news_df = (
            news_df
            .drop_duplicates(
                subset=["Headline"]
            )
            .reset_index(
                drop=True
            )
        )

        return news_df

    except Exception as e:

        print(
            f"News Pipeline Error: {e}"
        )

        return pd.DataFrame()