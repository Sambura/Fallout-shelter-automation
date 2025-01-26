# Few words about loot collection autotests

 * Each test is in a directory named `XXX-[G][C][N]`, where XXX - ordinal test number, G - if contains generic loot, C - corpse loot, N - neither
 * Each directory contains 3 .png files: 01-frame.png, 02-frame.png as source frames, and autotest-frame.png containing correct loot locations

## Autotest frame

Autotest frame should have exactly the same dimensions as the source frames. It should be black everywhere except the locations
where the loot must be detected. It is allowed to use any of 3 colors (#ff0000, #00ff00, #0000ff) to mark loot location. Each lootable object
should be represented with **one** continuous fragment painted with one of the colors above. Thus, it is not allowed to have multiple fragments
representing the same object. The fragment should have hard edges, i.e. no smoothed colors on the edges. If two loot objects overlap in some way, 
use two different colors to paint each of them, and the combination of these colors in the area where they overlap (e.g. #ff0000 + #0000ff = 
#ff00ff). If the two objects overlap too much to where it may be difficult to distinguish between them, you could treat such objects as a single
lootable object. Each fragment should also contain a symbol identifying it as generic loot vs. corpse loot. The symbol should be painted in any 
color other than any of the base colors or their combinations (i.e. all channels should be below 255). The symbol should have a closed border,
with distinct inner and outer areas. Mark generic loot with a square, and corpse loot with a equilateral-ish triangle.
