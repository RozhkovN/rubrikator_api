import type { Rubric } from "../types/chat";

export const formatTemplateText = (text: string) => {
  return text
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => line.trim());
};

export const formatResults = (
  rubric: Rubric | undefined,
  allRubrics: Rubric[] | null,
) => {
  if (!rubric) {
    return "Нет данных для отображения.";
  }

  let result = `Название: ${rubric.rubric_name}\n`;
  result += `Краткое название: ${rubric.short_name}\n`;
  result += `Уверенность: ${(rubric.confidence * 100).toFixed(2)}%\n\n`;
  result += `Ответ:\n${rubric.response_template}`;

  if (allRubrics && allRubrics.length > 1) {
    result += "\n\nВсе варианты:\n";
    allRubrics.forEach((r, index) => {
      result += `${index + 1}. ${r.short_name} (${(r.confidence * 100).toFixed(
        2,
      )}%)\n`;
    });
  }

  return result;
};
