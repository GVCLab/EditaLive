# EditaLive — Supported Edits

EditaLive edits a person in a video while keeping their identity, pose and motion unchanged. Based on training data, this page lists **what you can ask it to do**: which objects it can add, which it can remove, which it can recolour, and which artistic styles it can apply.

An edit prompt describes **one** change, in plain English, naming the object and where it is on the body — for example `Add a dark red scarf around his neck.`


## Quick reference

| Edit | Prompt template | Example |
|---|---|---|
| **Add** | `Add a <colour> <object> <location>.` | `Add a black fedora on his head.` |
| **Remove** | `Remove the <colour> <object> from <location>.` | `Remove the black fedora from his head.` |
| **Change** (colour) | `Change <his/her/the> <object> [on <location>] to <colour>.` | `Change her hoodie to light gray.` / `Change her hair to blonde.` |
| **Style** | Use one of the style prompts below verbatim. | `Convert it into pixel art style with limited resolution and color palette.` |

## Add

EditaLive can put any of these onto the person. Say the colour you want and where it goes; the object will follow the person's head and body motion.

**Hats & headwear**

| Object | Example prompt |
|---|---|
| Bridal floral headpiece | `Add a pink and cream floral headpiece on top of her head.` |
| Crown | `Add a purple crown on his head.` |
| Sports headband | `Add a black sports headband with a red stripe across his forehead.` |
| Newsboy cap | `Add a brown newsboy cap on her head.` |
| Bucket hat | `Add a white bucket hat on her head.` |
| Fedora | `Add a gray fedora with a red band on his head.` |
| Graduation cap | `Add a black graduation cap with a yellow tassel on his head.` |
| Knitted beanie | `Add a dark green knitted beanie on his head.` |
| Baseball cap | `Add a brown baseball cap on his head.` |

**Eyewear & face**

| Object | Example prompt |
|---|---|
| Sports goggles | `Add black sports goggles on his face.` |
| Sunglasses | `Add pink frame sunglasses to his face.` |
| Eyeglasses | `Add blue frame eyeglasses on his face.` |
| Masquerade mask | `Add a gold masquerade mask on her face.` |

**Earrings & headphones**

| Object | Example prompt |
|---|---|
| Over-ear headphones | `Add white over-ear headphones on her head.` |
| Dangle earrings | `Add blue dangle earrings to her ears.` |
| Hoop earrings | `Add large silver hoop earrings to both ears.` |

**Neck & chest**

| Object | Example prompt |
|---|---|
| Scarf | `Add a dark red scarf around his neck.` |
| ID badge lanyard | `Add a dark red ID badge lanyard with a white badge around her neck.` |
| Bowtie | `Add a dark blue bowtie at her neck.` |
| Necktie | `Add a dark red necktie around his neck.` |
| Brooch | `Add a green and silver brooch to the left side of his chest.` |
| Necklace | `Add a gold necklace with a green pendant around her neck.` |
| Boutonniere | `Add a white boutonniere to his left lapel.` |

**Wrist**

| Object | Example prompt |
|---|---|
| Smartwatch | `Add a black smartwatch on his right wrist.` |
| Digital watch | `Add a black digital watch to his left wrist.` |
| Sports wristband | `Add a brown sports wristband to his right wrist.` |
| Bracelet | `Add a silver bracelet to her raised wrist.` |


## Remove

EditaLive can take any of these off the person and reconstruct what was underneath. The list mirrors **Add** — anything it can add, it can also remove.

**Hats & headwear**

| Object | Example prompt |
|---|---|
| Bridal floral headpiece | `Remove the pink and cream floral headpiece from the top of her head.` |
| Crown | `Remove the purple crown from his head.` |
| Sports headband | `Remove the black sports headband with a red stripe from his forehead.` |
| Newsboy cap | `Remove the brown newsboy cap from her head.` |
| Bucket hat | `Remove the white bucket hat from her head.` |
| Fedora | `Remove the gray fedora with a red band from his head.` |
| Graduation cap | `Remove the black graduation cap with a yellow tassel from his head.` |
| Knitted beanie | `Remove the dark green knitted beanie from his head.` |
| Baseball cap | `Remove the brown baseball cap from his head.` |

**Eyewear & face**

| Object | Example prompt |
|---|---|
| Sports goggles | `Remove the black sports goggles from his face.` |
| Sunglasses | `Remove the pink frame sunglasses from his face.` |
| Eyeglasses | `Remove the blue frame eyeglasses from his face.` |
| Masquerade mask | `Remove the gold masquerade mask from her face.` |

**Earrings & headphones**

| Object | Example prompt |
|---|---|
| Over-ear headphones | `Remove the white over-ear headphones from her head.` |
| Dangle earrings | `Remove the blue dangle earrings from her ears.` |
| Hoop earrings | `Remove the silver hoop earrings from both ears.` |

**Neck & chest**

| Object | Example prompt |
|---|---|
| Scarf | `Remove the dark red scarf from his neck.` |
| ID badge lanyard | `Remove the dark red ID badge lanyard from her neck.` |
| Brooch | `Remove the green and silver brooch from the left side of his chest.` |
| Necktie | `Remove the dark red necktie from his neck.` |
| Necklace | `Remove the gold necklace with the green pendant from her neck.` |
| Bowtie | `Remove the dark blue bowtie from her neck.` |
| Boutonniere | `Remove the white boutonniere from his left lapel.` |

**Wrist**

| Object | Example prompt |
|---|---|
| Smartwatch | `Remove the black smartwatch from his right wrist.` |
| Digital watch | `Remove the black digital watch from his left wrist.` |
| Sports wristband | `Remove the brown sports wristband from his right wrist.` |
| Bracelet | `Remove the silver bracelet from her raised wrist.` |


## Change — colour

Recolouring keeps the edited object's shape, texture and motion and changes only its colour. It supports hair, garments and accessories; naming the current colour is optional, but it helps when several similar objects are visible.

**Hair**

| Object | Example prompt |
|---|---|
| Hair | `Change her hair to blonde.` |

**Hats & headwear**

| Object | Example prompt |
|---|---|
| Cap | `Change his cap to brown.` |
| Hat | `Change his hat to white.` |

**Eyewear & face**

| Object | Example prompt |
|---|---|
| Eyeglasses | `Change the eyeglasses on her face to dark purple.` |
| Sunglasses | `Change the sunglasses on her head to black.` |

**Earrings & headphones**

| Object | Example prompt |
|---|---|
| Headphones | `Change his headphones to black.` |

**Neck & chest**

| Object | Example prompt |
|---|---|
| Necklace | `Change the dark gray necklace pendant on her neck to blue.` |
| Tie | `Change his tie to dark navy.` |

**Wrist**

| Object | Example prompt |
|---|---|
| Watch | `Change the watch on her right wrist to silver and tan.` |
| Bracelet | `Change the bracelet on his left wrist to black.` |

**Upper body**

| Object | Example prompt |
|---|---|
| Hoodie | `Change the mustard yellow hoodie on her upper body to dark gray.` |
| Sweater | `Change the light purple sweater on her upper body to light gray.` |
| Jacket | `Change the teal jacket on her upper body to black.` |
| Top | `Change the pale yellow top on her torso to gray.` |
| Dress | `Change the deep blue dress on her body to light blue.` |
| Shirt | `Change the pink shirt on his upper body to light blue.` |
| Cardigan | `Change the orange cardigan on her upper body to gray.` |
| Blazer | `Change his blazer to brown.` |
| Blouse | `Change her blouse to black.` |
| Sweatshirt | `Change the dark burgundy sweatshirt on his upper body to light blue.` |

**Lower body**

| Object | Example prompt |
|---|---|
| Pants | `Change the light purple pants to dark black pants on her legs.` |
| Jeans | `Change the red jeans on his legs to blue.` |
| Shorts | `Change the red shorts to olive green on his lower body.` |
| Leggings | `Change the teal leggings on her legs to black.` |

**Footwear**

| Object | Example prompt |
|---|---|
| Sandals | `Change the yellow sandals on her feet to white.` |
| Shoes | `Change the bright pink shoes at her feet to tan.` |
| Sneakers | `Change the bright pink sneakers on her feet to light gray.` |
| Boots | `Change the brown boots on his feet to black.` |

EditaLive changes **colour, not pattern or material**. Asking for plaid, stripes, camouflage, a logo, or a leather-to-denim swap will not work reliably.



## Style

Style transfer re-renders the whole subject in an artistic style, rather than editing one object. Use these prompts verbatim for the most predictable result.

| Style | Prompt |
|---|---|
| American comic book | `Transform it into an American comic book style rendering with bold outlines, clean inking, and stylized shading.` |
| Anime (2D cel-shaded) | `Transform it into an anime-style 2D rendering with cel shading, clean line art, and stylized features.` |
| C4D felt doll | `Turn it into a C4D felt doll style.` |
| Chinese 3D anime | `Transform it into a Chinese 3D anime style.` |
| Chinese classical animation | `Turn it into a refined traditional Chinese decorative animation illustration inspired by classic Chinese animated films.` |
| Chinese ink wash painting | `Transform it into a traditional ink wash painting style with brush strokes and monochrome gradients.` |
| Classical oil painting | `Re-render it in a classical oil painting style with visible brush strokes and rich textures.` |
| Claymation / clay toy | `Transform it into a clay toy (claymation) style with soft plastic textures, rounded forms, and handcrafted details.` |
| Disney 2D animation | `Transform it into a 2D Disney-style animated rendering with soft lighting, expressive features, and polished shading.` |
| Family Guy | `Transform it into a Family Guy style.` |
| Flat vector illustration | `Convert it into a clean vector illustration style with flat colors, sharp edges, and minimal shading.` |
| Knitted yarn | `Convert it into a knitted yarn style with soft woven textures and fabric-like surfaces.` |
| Low-poly 3D | `Transform it into a low-poly 3D style with simplified geometric shapes and flat shading.` |
| Pixar-style 3D | `Re-render it in a Pixar-style 3D animation look with smooth shading, soft global illumination, and vibrant colors.` |
| Pixel art | `Convert it into pixel art style with limited resolution and color palette.` |
| Plush toy | `Transform it into a soft plush-like aesthetic with smooth textures and gentle shading.` |
| Rick & Morty cartoon | `Re-render it in the style of Rick and Morty with bold outlines, flat colors, and simplified cartoon features.` |
| Rubber hose (black & white) | `Re-render it in a classic black-and-white rubber hose animation style.` |
| Snoopy / Peanuts | `Transform it into a Snoopy style with simple clean outlines and flat colors.` |
| Studio Ghibli | `Re-render it in a Studio Ghibli-inspired hand-painted animation style with soft lighting and rich textures.` |
| Unreal Engine render | `Convert it into an Unreal Engine-style real-time render with high-quality lighting and materials.` |
