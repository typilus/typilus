CREATE TEMP FUNCTION check4Typing(arr ARRAY<STRING>)
RETURNS BOOL
LANGUAGE js AS """
  return arr && arr.some(function(ele) {
    return ele.startsWith("import typing") || ele.startsWith("from typing import") || ele.includes("# type: ") ;
  });
  """;

WITH typedRepos AS (
  SELECT sample_repo_name AS repo
  FROM `fh-bigquery.github_extracts.contents_py`
  WHERE check4Typing(SPLIT(content, "\n"))
  GROUP BY repo
)

SELECT repo
FROM typedRepos INNER JOIN `fh-bigquery.github_extracts.repo_stars` AS repoStars
ON typedRepos.repo = repoStars.repo_name
ORDER BY stars DESC
LIMIT 1000;

