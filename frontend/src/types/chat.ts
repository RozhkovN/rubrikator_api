export interface Rubric {
  rubric_id: number;
  rubric_name: string;
  short_name: string;
  response_template: string;
  confidence: number;
}

export interface APIResponse {
  text: string;
  best_match: Rubric;
  all_predictions: Rubric[] | null;
}

export interface FileResultWithOptions {
  filename: string;
  text: string;
  best_match: Rubric;
  all_predictions: Rubric[] | null;
  error: string | null;
}

export interface SimpleFileResult {
  filename: string;
  text: string;
  rubric_id: number;
  rubric_name: string;
  short_name: string;
  response_template: string;
  confidence: number;
  error: string | null;
}

export interface MultiFileResponse {
  results: (FileResultWithOptions | SimpleFileResult)[];
  total: number;
  success: number;
  failed: number;
}

export type Message = {
  id: string;
  text: string;
  isUser: boolean;
  isLoading?: boolean;
  rubric?: Rubric;
  allRubrics?: Rubric[] | null;
  error?: string;
  fileName?: string;
  isFileMessage?: boolean;
  fileResults?: (FileResultWithOptions | SimpleFileResult)[];
  showTemplate?: boolean;
};

export type ExpandedRubric = {
  [rubricId: number]: boolean;
};

export type ExpandedFileRubric = {
  [key: string]: boolean;
};

export type FileTemplateVisibility = {
  [fileIndex: number]: boolean;
};
