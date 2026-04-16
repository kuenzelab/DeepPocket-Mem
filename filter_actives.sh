#!/bin/sh
awk '
  $1 == 1 {
    base = $5
    sub("_nowat.*", "", base)
    seen[base] = 1
  }
  {
    base = $5
    sub("_nowat.*", "", base)
    if (seen[base]) print $0
  }
' all.types > all_with_actives.types
