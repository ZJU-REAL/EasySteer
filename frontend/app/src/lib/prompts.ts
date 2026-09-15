/** Render the single-instruction format saved in native training checkpoints. */
export function renderPrompt(template: string, instruction: string): string {
  let placeholders = 0;
  const prompt = template.replace(/%%|%s|%/g, (token) => {
    if (token === "%%") return "%";
    if (token === "%s") {
      placeholders += 1;
      return instruction;
    }
    throw new Error("Prompt template must use one %s placeholder and %% for literal percent signs");
  });
  if (placeholders !== 1) throw new Error("Prompt template must contain one %s placeholder");
  return prompt;
}
