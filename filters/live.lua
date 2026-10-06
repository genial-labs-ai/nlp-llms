-- Marks what a lecture teaches in the room (its `live` plan in the front matter,
-- scripts/live_plan.py) so the page shows it. It never changes the text of a page.
--
--   1. A block (`::: {#id .self-check}`, `.demo`, `.predict`, `.discuss`) named in the
--      plan's activities gets the class `live`: custom.scss gives it an "In the room" badge.
--   2. A numbered section heading with the class `reference` is not taught in the room;
--      custom.scss gives it a "Reference" badge. Nothing is hidden.
--
-- Two passes: pandoc visits Meta after the blocks, so the plan is read first.

local stringify = pandoc.utils.stringify
local planned = {}

local function read_plan(meta)
  if not meta.live then
    return nil
  end
  for _, entry in ipairs(meta.live) do
    if entry.activities then
      for _, activity in ipairs(entry.activities) do
        planned[stringify(activity.block)] = true
      end
    end
  end
  return nil
end

local function mark(div)
  if div.identifier ~= "" and planned[div.identifier] then
    div.classes:insert("live")
    return div
  end
end

return {
  { Meta = read_plan },
  { Div = mark },
}
