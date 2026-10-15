TextCraft is a text-based crafting game built from Minecraft's crafting recipes. Each task names one goal item and lists the crafting commands available in that task. The task is won the moment the goal item enters the inventory; there is no "done" or "submit" command.

**Item types.** Raw materials are gathered directly from the environment with `get`, in any quantity; they cannot be crafted. Intermediate items are crafted from raw materials or from other intermediate items. The goal item may need a chain of intermediate items before it can be crafted. Only raw materials can be gathered: an item that can be crafted cannot be obtained with `get`.

**Crafting commands.** Each listed command has the form `craft <number> <item> using <number> <ingredient>, <number> <ingredient>, ...`, for example `craft 4 oak planks using 1 oak logs`. Running a command once removes exactly the listed ingredients from the inventory and adds the stated number of the crafted item; to make more, run it again. Only the listed commands can be used, with exactly the listed numbers (the order of the ingredients does not matter). Every ingredient must be in the inventory before crafting; if one is missing, nothing changes.

**Ingredient categories.** Some ingredients, such as `planks` or `wool`, are categories: any item of the category satisfies them. The task lists the items of every category it uses. In a crafting command you may write the category (held items of that category are used) or one specific item of it, for example `2 oak planks` instead of `2 planks`.

**Names.** Item names are lowercase words separated by spaces, written exactly as in the listed commands (`dark oak planks`). Names are taken literally, including singular and plural: `oak logs` is not `oak log`.

**Commands.**
- `get <number> <item>`: gather a raw material, in any amount, for example `get 2 oak logs`.
- `craft <crafting command>`: run one of the listed crafting commands once.
- `inventory`: list the held items and their quantities. The inventory starts empty.

**Strategy.** Plan before acting. Find the listed command that crafts the goal item, then work backwards through its ingredients, noting which must be crafted and which are raw, down to raw materials. Total how many of each raw material the whole chain needs: a command that makes 4 items consumes its ingredients once, so count how many times each command has to run. Then gather all the raw materials in sufficient quantities, craft the intermediate items in dependency order (an item only after everything it needs), and craft the goal item last.

**Common mistakes.** Crafting before gathering everything a command needs; crafting in the wrong order; misspelled item names; gathering too few raw materials for the whole chain; forgetting that crafting consumes its ingredients.

**Budget.** Every tool call counts toward the episode's tool-call budget, including failed and invalid commands.
